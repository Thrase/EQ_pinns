import torch
import matplotlib.pyplot as plt
from tqdm import trange

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Feedforward(torch.nn.Module):
    """
    A class for defining feed-forward neural networks.

    Attributes
    ----------
    input_size : int
        Sets the dimension of the problem domain.
    hidden_size : int
        Sets the number of neurons to be used in each layer of the network.
    ϕ : activation
        A component-wise activation function applied to each network layer.
    linear_i : linear
        Applies a linear transformation to data from the i-1 layer.
        This transformation is given by y=xAᵀ + b for network weight A and
        bias b on input x.
    opt : optim
        An object inherited from the Module class that is used to set the
        optimization algorithm to be used during training.
    n : float
        A number used to scale adaptive weights
    a : parameter
        A trainable network parameter that scales the weight matrices to
        improve training times.

    Methods
    -------
    forward(x):
        Computes the forward pass of the network using input x.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize an instance of FeedForward.

        Arguments
        ----------
        input_size : int
            Sets the dimension of the problem domain.
        hidden_size : int
            Sets the number of neurons to be used in each layer of the network.
        """
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_1 = torch.nn.Linear(self.input_size, self.hidden_size,
                                        bias=True)
        self.linear_2 = torch.nn.Linear(self.hidden_size, self.hidden_size,
                                        bias=True)
        self.linear_3 = torch.nn.Linear(self.hidden_size, 1, bias=True)

        self.ϕ = torch.nn.Sigmoid()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01,
                                    betas=(0.9, 0.999), foreach=True)

        self.n = 1.0  # adaptive  weight scaling
        self.a = torch.nn.Parameter(torch.ones(1) / self.n)
        self.b = torch.nn.Parameter(torch.ones(1) / self.n)

    def forward(self, x):
        """
        Compute the forward pass of the network on input x.

        Adaptive weight scaling makes a slight adjustment to the computation at
        each layer. Namely, the iᵗʰ layer computation becomes
        na ϕ.(Wᵢ xᵢ₋₁ + bᵢ).

        Arguments
        ----------
        x : Tensor
            A tensor of shape [1,k] where k is the dimension of the problem
            domain.
        """
        hidden = self.linear_1(x)
        hidden = self.n * self.a * self.ϕ(hidden)
        output = self.linear_2(hidden)
        output = self.n * self.b * self.ϕ(output)
        output = self.linear_3(output)
        return output


num_iter = 10000
batch_size = 200
hidden_dim = 150

x0, x1 = [0.0, 1.0]
t0, t1 = [0.0, 1.0]

Net = Feedforward(2, hidden_dim).to(device)


def NN(z):
    """
    Define a trial function that gives hard enforcement of I.C.

    Using the initial data for displacement and velocity we construct
    a function which inherently satisfies the initial state of the system.
    The trial function takes the form NN(z) = u₀ + tv₀ + t²N(z) where u₀
    and v₀ are initial data for displacement and velocity, respectively.
    The trainable portion of the network is given by the neural net N.
        Parameters:
            z (tensor): A tensor representation of a point in the domain.
        Returns:
            The neural net approximate solution to the IBVP.
    """
    x, t = z.split(1, dim=1)
    t0 = torch.zeros(t.shape, requires_grad=True)
    z0 = torch.cat([x, t0], dim=1).to(device)

    return uexact(z0) + uexact_t(z0)*t + Net(z)*(t**2)


def uexact(z):
    """
    Define a function to act as a manufactured solution.

    Arguments
    ----------
    z  : tensor
        A tensor representation of a point in the domain.
    """
    x, t = z.split(1, dim=1)
    return torch.sin(2*torch.pi*(x - t))


def uexact_t(z):
    """
    Define an initial velocity based on the exact solution.

    Arguments
    ----------
    z : tensor
        A tensor representation of a point in the domain.
    """
    x, t = z.split(1, dim=1)
    return -2*torch.pi * torch.cos(2*torch.pi*(x-t))


def velocity(z):
    """
    Define the network velocity function.

    Compute the time derivative of the network using the PyTorch automatic
    differentiation engine.

        Parameters:
            z (tensor): A tensor representation of a point in the domain.
        Returns:
            dt (tensor): Automatic time derivative of the neural network.
    """
    grad, = torch.autograd.grad(NN(z).sum(), z, create_graph=True)
    dx, dt = grad.split(1, dim=1)
    return dt


def wave_relation(z):
    """
    Define the neural net wave equation.

    Using autograd to compute network derivatives w.r.t space and time
    lets us define the neural net wave equation. The result is a new network
    which outputs Nₜₜ - c²Nₓₓ.

        Parameters:
            z (tensor): A tensor representation of a point in the domain.
        Returns:
            dtt-dxx (tensor): The network wave equation.
    """
    # use autograd to get first-order network derivatives
    grad, = torch.autograd.grad(NN(z).sum(), z, create_graph=True)
    dx, dt = grad.split(1, dim=1)

    # another call to autograd yields second-order derivatives
    grad_dx, = torch.autograd.grad(dx.sum(), z, create_graph=True)
    grad_dt, = torch.autograd.grad(dt.sum(), z, create_graph=True)
    dxx, dxt = grad_dx.split(1, dim=1)
    dtx, dtt = grad_dt.split(1, dim=1)
    return dtt - dxx


def my_loss(z, z_x0, z_x1):
    """
    Define loss function to be optimized.

    The loss function is assembled using the criteria specified by the IBVP
    we wish to solve.
        Parameters:
            z (tensor): Interior point of the domain for constraining the PDE.
            z_x0 (tensor): Boundary point at x=0 for constraining B.C.
            z_x1 (tensor): Boundary point at x=1 for constraining B.C.
        Returns:
            L (tensor): Sum of system mean square errors.
    """
    L = wave_relation(z).abs().square().sum() / z.shape[0]
    L += (NN(z_x0) - uexact(z_x0)).abs().square().sum() / z_x0.shape[0]
    L += (NN(z_x1) - uexact(z_x1)).abs().square().sum() / z_x1.shape[0]
    return L


# begin network training iterations:
for i in trange(num_iter):
    x = torch.rand((batch_size, 1), requires_grad=True)
    t = torch.rand((batch_size, 1), requires_grad=True)
    z = torch.cat([x, t], dim=1).to(device)

    x = x0 * torch.ones((batch_size, 1), requires_grad=True)
    t = torch.rand((batch_size, 1), requires_grad=True)
    z_x0 = torch.cat([x, t], dim=1).to(device)

    x = x1 * torch.ones((batch_size, 1), requires_grad=True)
    t = torch.rand((batch_size, 1), requires_grad=True)
    z_x1 = torch.cat([x, t], dim=1).to(device)

    # Zero your gradients for every batch!
    # for param in net_term.parameters():
    #    param.grad = None
    Net.opt.zero_grad()

    # Compute the loss and its gradients
    loss = my_loss(z, z_x0, z_x1)
    loss.backward()

    # Adjust learning weights
    Net.opt.step()

x_grid = torch.arange(x0, x1, 0.01).unsqueeze(1)
t_grid = torch.ones(x_grid.shape)
z = torch.cat([x_grid, t_grid], dim=1).to(device)

U = NN(z).detach().numpy()
U2 = uexact(z).detach().numpy()

plt.plot(U)
plt.plot(U2)
plt.show()
