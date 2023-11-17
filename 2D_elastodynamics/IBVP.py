from geometry_2D import *
import torch
import StrikeSlipMMS as MMS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

μ = 32

ρ =  2.670
c = np.sqrt(μ / ρ)  # wave speed
Z = np.sqrt(μ * ρ)  # shear impedance
a = 0.015
b = 0.02 
L =  2.0  # meters
f0 =  0.6
σn =  50.0
v0 =  10e-6  # m/s

mms = MMS.StrikeSlipMMS()

mse = torch.nn.MSELoss()
def friction_coeff(z):
    x, y, t = z.split(1, dim=1)
    
    
    y1 = torch.where(y < 12, -0.005, 0.0)
    y2 = torch.where(y > 17, 0.015, 0.0)
    y3 = torch.where((y >= 12) & (y <= 17), (y - 12) * ((0.015+0.005)/5) - 0.005, 0.0)
    out = y1+y2+y3
        
    return out
class Condition:
    """A class for automating tasks used to train the PINN loss function."""

    def __init__(self, domain, data_function=None, t_initial=0.0, t_final=1.0):
        self.domain = domain
        self.data_function = data_function
        self.test_data = None
        self.t_initial = t_initial
        self.t_final = t_final
        self.prev_loss = [torch.tensor(0.0)]
        self.sampled_points = None
        
    def sample_domain(self, n_samples=1, uniform=False):
        """Generate input data on a subdomain and generate test data."""
        
        pt = self.domain.get_point(n_samples, uniform=uniform)
        τ = torch.ones((n_samples, 1)).uniform_(
            self.t_initial, self.t_final)
 
        pt = torch.cat([pt, τ], dim=1)

        if self.data_function is not None:
            self.test_data = self.data_function(pt).to(device)
        
        self.sampled_points = pt.to(device)
        
        return pt.to(device)

    def loss(self):
        """Compute the error given of a condition function w.r.t test_data."""
        
        output = self.condition(self.sampled_points)
        loss = mse(output, self.test_data)
        self.prev_loss.append(loss.detach())
        return loss


class BoundaryCondition(Condition):
    def __init__(self, domain, NNet=None, data_function=None, bc='dirichlet'):
        #BoundaryCurve.__init__(self, P)
        Condition.__init__(self, domain, data_function)

        self.bc = bc
        self.NNet = NNet

        if self.bc == 'dirichlet':
            self.condition = self.NNet

        elif self.bc == 'neumann':
            self.condition = self.neumann
    '''   
    def sample_bdry(self, t0=1.0):
        t = torch.rand((self.Nsamples, 1))
        z = self(t)
        τ = torch.rand((self.Nsamples, 1))
        z = torch.cat([z, τ], dim=1)
        
        self.input = z
        self.test_data = self.bdry_data(z)
        self.coordinate = t
    '''

    def neumann(self, z):
        z.requires_grad = True

        output = self.NNet(z)
        output = output.sum()
        grad, = torch.autograd.grad(output, z, create_graph=True)

        dx, dy, dt = grad.split(1, dim=1)
        
        n = self.domain.normal().to(device)
        nx, ny = n.split(1, dim=1)
        
        τ = nx*dx + ny*dy
        

        return μ * τ


class InitialCondition(Condition):
    def __init__(self, domain, NNet=None, data_function=None,
                 bc='primal'):
        Condition.__init__(self, domain, data_function)
        #self.domain = domain
        self.NNet = NNet
        if bc == 'primal':
            self.condition = self.NNet
        elif bc == 'velocity':
            self.condition = self.dt

    def dt(self, z):
        """Compute a network derivative w.r.t time."""
        z.requires_grad = True

        output = self.NNet(z)
        output = output.sum()
        grad, = torch.autograd.grad(output, z, create_graph=True)

        _, _, dt = grad.split(1, dim=1)  # gradient can be split into parts

        #loss = self.NNet.mse(dt, self.data_out)

        return dt

        
class Characteristic(Condition):
    def __init__(self, domain, NNet=None, data_function=None, R=0.0):
        Condition.__init__(self, domain, data_function)
        #self.domain = domain
        self.NNet = NNet
        self.R = R
        
    def characteristic(self, z, uexact=None):
        z.requires_grad = True
        
        if uexact is not None:
            output = uexact(z)
        else:
            output = self.NNet(z)
            
        output = output.sum()
        grad, = torch.autograd.grad(output, z, create_graph=True)
        
        dx, dy, dt = grad.split(1, dim=1)
        
        n = self.domain.normal().to(device)
        nx, ny = n.split(1, dim=1)
        
        τ = nx*dx + ny*dy
        
        src = Z*dt*(1 - self.R) + μ*τ*(1 + self.R)
        
        return src
        
    def condition(self, z):
        """Compute a network derivative w.r.t time."""
        #z.requires_grad = True

        #self.test_data = self.characteristic(z, uexact=self.data_function).detach()
        
        output = self.characteristic(z)
        
        return output
        

class RateFault(Condition):
    def __init__(self, domain, NNet=None, fric_net=None, data_function=None):
        Condition.__init__(self, domain, data_function)
        #self.domain = domain
        self.NNet = NNet
        self.fric_net = fric_net
        

        
    def friction(self, z):
        eps = torch.tensor([10e-12]).to(device)
        z.requires_grad = True # = x.clone().detach().requires_grad_(True)
        _, depth, _ = z.split(1, dim=1)
        #z.requires_grad = True
        
        #if (uexact is not None) and (sexact is not None):
        #u_out = uexact(z)
        #s_out = sexact(z)
        #else:
        u_out = self.NNet(z)
        a = self.fric_net(depth) # friction_coeff(z)
            
        u_out = u_out.sum()
        grad, = torch.autograd.grad(u_out, z, create_graph=True)
        
        dx, dy, dt = grad.split(1, dim=1)
        
        fault_str = f0 + a*torch.log((2*dt.abs() / (v0)))
        
        #alternate = 2 * v0 * torch.exp((μ*dx / (σn*(a-b))) - (f0 / (a-b))) - dt
        #fault = a * torch.log((v0/dt.abs()) * torch.sinh((-μ*τ).abs() / (a*50.0))) 
        #state = s_out - fault
        fault = (-μ * dx) - σn*fault_str
        return fault       
                        
    def condition(self, z):
        """Compute a network derivative w.r.t time."""
        #z.requires_grad = True
        output = self.friction(z)
        
        return output
                

class PDE(Condition):
    def __init__(self, domain, NNet=None, data_function=None):
        Condition.__init__(self, domain, data_function=data_function)
        self.NNet = NNet
        #self.domain = domain
        #self.ti, self.tf = self.domain.t_bounds
        #self.tf = self.domain.tf

    def condition(self, z):
        z.requires_grad = True

        u = self.NNet(z)
        u = u.sum()

        grad, = torch.autograd.grad(u, z, create_graph=True)
        dx, dy, dt = grad.split(1, dim=1)  # gradient can be split into parts

        grad_dx, = torch.autograd.grad(dx.sum(), z, create_graph=True)
        grad_dy, = torch.autograd.grad(dy.sum(), z, create_graph=True)
        grad_dt, = torch.autograd.grad(dt.sum(), z, create_graph=True)

        dxx, dxy, dxt = grad_dx.split(1, dim=1)
        dyx, dyy, dyt = grad_dy.split(1, dim=1)
        dtx, dty, dtt = grad_dt.split(1, dim=1)

        Δ = dtt - (c**2) * (dxx + dyy)
        return Δ

        
        
class IBVP:
    def __init__(self, domain):
        self.BC = {}
        self.domain = domain
        self.IC = {}
        self.PDE = {}
        
            
    def set_bc(self, NNet=None, data_function=None, bc='dirichlet', label=None):
        self.BC[label] = BoundaryCondition(self.domain.bdry[label], NNet=NNet,data_function=data_function, bc=bc)
        
        
    def set_ic(self, NNet=None, data_function=None, bc='primal', label=None):
        self.IC[label] = InitialCondition(self.domain, NNet=NNet, data_function=data_function, bc=bc)

    def loss_report(self):
        str_list = []
        str_list.append("╔" + "{:═^10s}".format("") + "╤" + "{:═^12s}".format("")    +"╗")
        str_list.append("║" + "{:^10s}".format("")  + "│" + "{:^12s}".format("Loss") +"║") 
        
        for key, val in self.PDE.items():
            str_list.append("╠" + "{:═^10s}".format("") + "╪" + "{:═^12s}".format("") + "╣")
            str_list.append("║" + "{:^10s}".format(key) + "│" + "{:^12.3e}".format(val.loss()) +"║")
            
        for key, val in self.BC.items():
            str_list.append("╠" + "{:═^10s}".format("") + "╪" + "{:═^12s}".format("") + "╣")
            str_list.append("║" + "{:^10s}".format(key) + "│" + "{:^12.3e}".format(val.loss()) +"║")
        
        for key, val in self.IC.items():
            str_list.append("╠" + "{:═^10s}".format("") + "╪" + "{:═^12s}".format("") + "╣")
            str_list.append("║" + "{:^10s}".format(key) + "│" + "{:^12.3e}".format(val.loss()) +"║")
        
            
        str_list.append("╚" + "{:═^10s}".format("") + "╧" + "{:═^12s}".format("") + "╝")
        
        return np.asarray(str_list)
        
    def network_report(self, net_dict):
        str_list = []
        count = 0
        
        for name,net in net_dict.items():
            str_list.append([])
            str_list[count].append(" ╟" + "{:─^19s}".format(name) +"╢ ")
            str_list[count].append(" ╠" + "{:═^9s}".format("") + "╦" + "{:═^9s}".format("")    +"╣ ")
            str_list[count].append(" ║" + "{:^9s}".format("max|w∇|")  + "║" + "{:^9s}".format("max|b∇|") +"║ ")
        
            for layer in net.linears:
                max_wgrad = layer.weight.grad.abs().max()
                max_bgrad = layer.bias.grad.abs().max()
            
                str_list[count].append(" ╠" + "{:═^9s}".format("") + "╬" + "{:═^9s}".format("")    +"╣ ")
                str_list[count].append(" ║" + "{:^9.2e}".format(max_wgrad)  + "║" + "{:^9.2e}".format(max_bgrad) +"║ ")
            
            str_list[count].append(" ╚" + "{:═^9s}".format("") + "╩" + "{:═^9s}".format("") + "╝ ")
            
            count += 1
        
        
        max_table = np.max([len(k) for k in str_list])
        
        for k in str_list:
            diff = max_table - len(k)
            k.extend(["" for i in range(diff)])
        
        for i in range(len(str_list)):
            str_list[i] = np.asarray(str_list[i])  
        
            
        x = str_list[0] 
        
        for i in range(1,len(str_list)):
            x = np.char.add(x, str_list[i]) 
            
        #for i in x:
        #    print(i)
            
        return x
            
        
                
        
        
        
#print("╠" + "{:═^10s}".format("") + "╪" + "{:═^12s}".format("") + "╣")
#print("║" + "{:^10s}".format("surface") + "│" + "{:^12.3e}".format(IBVP.BC['fault'].loss()) +"║")




        
   
    
               
        

