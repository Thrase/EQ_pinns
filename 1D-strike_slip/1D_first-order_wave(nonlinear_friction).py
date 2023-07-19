import torch 
import numpy as np
from tqdm import trange
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

num_iter = 1000
batch_size = 100
hidden_dim = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.set_default_dtype(torch.float64)

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)    
        self.fc3 = torch.nn.Linear(self.hidden_size, 1, bias=True)
        
        self.sigmoid = torch.nn.Tanh()    
        
        
        self.n = 5.0  # adaptive weight scaling
        self.a = torch.nn.Parameter(torch.ones((1, hidden_size), requires_grad=True) / self.n)  # adaptive weights a, b
        self.b = torch.nn.Parameter(torch.ones((1, hidden_size), requires_grad=True) / self.n)
        self.Wa = torch.nn.Parameter(torch.rand((1,1), requires_grad=True))
        
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01, betas=(0.9, 0.999), foreach=True)
        
    def forward(self, x):
        output =  self.fc1(x)
        output = self.sigmoid(self.n * self.a * output)
        #output = self.sigmoid(output)
      
        output = self.fc2(output)
        output = self.sigmoid(self.n * self.b * output)
        #output = self.sigmoid(output)
        output = self.fc3(output)
        return output
        
    def adapt(self):
        return 1 /((1/2)*(torch.exp(self.a.sum()/self.a.shape[1]) + torch.exp(self.b.sum()/self.b.shape[1])))
        
        
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

γ = 1.0
β = 1.0

#x = torch.rand((batch_size,1), requires_grad=True)
#t = torch.rand((batch_size,1), requires_grad=True)
#z = torch.cat([x, t], dim=1)


def wexact(z):
    x, t = z.split(1, dim=1)
    return 1 - torch.tanh(x - c*t + 1)
    
def wexact_t(z):
    x, t = z.split(1, dim=1)
    return c * (1 - torch.tanh(x - c*t + 1 )**2)
    
def wexact_x(z):
    x, t = z.split(1, dim=1)
    return -(1 - torch.tanh(x - c*t + 1 )**2)

def w0(z):
    x, _ = z.split(1, dim=1)
    return 1 - torch.tanh(x + 1)
    
def v0(z):
    x, _ = z.split(1, dim=1)
    return c * (1 - torch.tanh(x + 1)**2)

# Now, define NNets for solution u and state variable ψ
# establish necessary functions to be expressed in terms of out NNets
w_net = Feedforward(2,hidden_dim).to(device)  # displacement network

def w(z):
    x, t = z.split(1, dim=1)
    return w0(z) + v0(z)*t + w_net(z)*(t**2)

def u(z):
    grad_w, = torch.autograd.grad(w(z).sum(), z, create_graph=True)  # gradient of network w.r.t x,y
    u, _ = grad_w.split(1,dim=1)  # gradient can be split into parts
    
    return u
    
def v(z):
    grad_w, = torch.autograd.grad(w(z).sum(), z, create_graph=True)  # gradient of network w.r.t x,y
    _, v = grad_w.split(1,dim=1)  # gradient can be split into parts
    
    return v

# This section we define functions for the IBVP 
def wave_relation(z):
    grad_w, = torch.autograd.grad(w(z).sum(), z, create_graph=True)  # gradient of network w.r.t x,y
    u, v = grad_w.split(1,dim=1)  # gradient can be split into parts
    
    
    grad_u, = torch.autograd.grad(u.sum(), z, create_graph=True)
    grad_v, = torch.autograd.grad(v.sum(), z, create_graph=True)
    
    ux, ut = grad_u.split(1, dim=1)
    vx, vt = grad_v.split(1, dim=1)
    
    l = (vt - (c**2) * ux).abs().square()
    l += (ut - vx).abs().square()
    return l
    

def fault_str(z):
    #return 2 * σn*v(z)
    return β*torch.arcsinh(2 * γ * v(z))
    
def fault_src(z):
    x, t = z.split(1, dim=1)
    #return (-μ - 2 * σn * c) * (1 - torch.tanh(x - c*t + 1)**2)
    return μ*wexact_x(z) - β*torch.arcsinh(2 * γ * wexact_t(z))
    
    
def my_loss(z, z_x0, z_x1, z_t01, z_t02):
    loss = wave_relation(z).sum() / z.shape[0]
    loss += (μ*u(z_x0) - fault_str(z_x0) - fault_src(z_x0)).abs().square().sum() / z_x0.shape[0]
    loss += (Z*v(z_x1) + μ*u(z_x1)).abs().square().sum() / z_x1.shape[0]
    
    #loss += (w(z_t01) - wexact(z_t01)).abs().square().sum() / z_t01.shape[0]
    #loss += (v(z_t02) - wexact_t(z_t02)).abs().square().sum() / z_t02.shape[0]
    return loss
 
x0, x1 = [0.0, 1.0]
t0, t1 = [0.0, 1.0]
   
for i in trange(num_iter):  

    x = torch.rand((batch_size,1), requires_grad=True)
    t = torch.rand((batch_size,1), requires_grad=True)
    z = torch.cat([x, t], dim=1).to(device)

    x = x0 * torch.ones((batch_size,1), requires_grad=True)
    t = torch.rand((batch_size,1), requires_grad=True)
    #t = torch.tensor(np.random.choice(t_grid, batch_size), requires_grad=True).unsqueeze(1)
    z_x0 = torch.cat([x, t], dim=1).to(device)

    x = x1 * torch.ones((batch_size,1), requires_grad=True)
    t = torch.rand((batch_size,1), requires_grad=True)
    #t = torch.tensor(np.random.choice(t_grid, batch_size), requires_grad=True).unsqueeze(1)
    z_x1 = torch.cat([x, t], dim=1).to(device)


    x = torch.rand((batch_size,1), requires_grad=True)
    #x = torch.tensor(np.random.choice(x_grid, batch_size), requires_grad=True).unsqueeze(1)
    t = t0*torch.ones((batch_size,1), requires_grad=True)
    z_t01 = torch.cat([x, t], dim=1).to(device)

    x = torch.rand((batch_size,1), requires_grad=True)
    #x = torch.tensor(np.random.choice(x_grid, batch_size), requires_grad=True).unsqueeze(1)
    t = t0*torch.ones((batch_size,1), requires_grad=True)
    z_t02 = torch.cat([x, t], dim=1).to(device)
    
    '''
    x = torch.zeros((batch_size,1), requires_grad=True)
    #x = torch.tensor(np.random.choice(x_grid, batch_size), requires_grad=True).unsqueeze(1)
    t = torch.zeros((batch_size,1), requires_grad=True)
    z_ψ0 = torch.cat([x, t], dim=1).to(device)
    '''
    x = torch.zeros((batch_size,1), requires_grad=True)
    #x = torch.tensor(np.random.choice(x_grid, batch_size), requires_grad=True).unsqueeze(1)
    t = torch.rand((batch_size,1), requires_grad=True)
    z_ψ = torch.cat([x, t], dim=1).to(device)

    
    # Zero network gradients for every batch!
    w_net.opt.zero_grad()
    #ψ_net.opt.zero_grad()
    #ψ.opt.zero_grad()

    # Make predictions for this batch
    #outputs = model(inputs)

    # Compute the loss and its gradients
    loss = my_loss(z, z_x0, z_x1, z_t01, z_t02)
    loss.backward()

    # Adjust learning weights
    w_net.opt.step()

  
t_grid = torch.arange(t0, t1, 0.005).requires_grad_().unsqueeze(1).to(device)
#t_grid = np.arange(t0, t1, 0.001)
x_grid = torch.arange(x0, x1, 0.005).requires_grad_().unsqueeze(1).to(device)
#t_grid = torch.zeros(x_grid.shape).requires_grad_().to(device)
z_grid = torch.cat([x_grid, t_grid], dim=1).to(device)

x2_grid = x_grid.cpu().detach().numpy()

wNN = []
wE = []
werr = []

uNN = []
uE = []
uerr = []

vNN = []
vE = []
verr = []

wNN = []
wE = []
werr = []

for i in range(len(t_grid)):
    t = torch.ones(x_grid.shape).to(device) * t_grid[i]
    #t = t.to(device)
    z_grid = torch.cat([x_grid, t], dim=1).to(device)
    w_a = w(z_grid)
    w_a = w_a.cpu().detach().numpy()
    wex = wexact(z_grid)
    wex = wex.cpu().detach().numpy()
    
    u_net = u(z_grid)
    u_net = u_net.cpu().detach().numpy()
    uexact = wexact_x(z_grid)
    uexact = uexact.cpu().detach().numpy()
    
    v_net = v(z_grid)
    v_net = v_net.cpu().detach().numpy()
    vexact = wexact_t(z_grid)
    vexact = vexact.cpu().detach().numpy()
    
    wNN.append(w_a)
    wE.append(wex)
    werr.append(w_a - wex)
    
    uNN.append(u_net)
    uE.append(uexact)
    uerr.append(u_net - uexact)
    
    vNN.append(v_net)
    vE.append(vexact)
    verr.append(v_net - vexact)
'''
p1 = w(z_grid)
p1 = p1.cpu().detach().numpy()

p2 = wexact(z_grid)
p2 = p2.cpu().detach().numpy()
t = t_grid.cpu().detach().numpy()
x = x_grid.cpu().detach().numpy()

plt.plot(x, p1, label="learned")
plt.plot(x, p2, label="exact")
plt.legend()
plt.show()
'''

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

Main.xfine = x2_grid
Main.tfine = t_grid

Main.wNet = wNN
Main.wExact = wE
Main.werr = werr

Main.uNet = uNN
Main.uExact = uE
Main.uerr = uerr

Main.vNet = vNN
Main.vExact = vE
Main.verr = verr

# visualize solution
Main.eval("""
using Plots

theme(:lime)    # the most powerful theme
color = theme_palette(:lime).colors.colors

anim = @animate for i=1:length(tfine)

  p1 = plot(xfine, wNet[i],color=color[1],
            lw=1.5, legend=:topright,ylims=(-0.0, 2.2),
            label = "network", xaxis=(""), linestyle=:dashdot,
            right_margin = 30Plots.mm, title="Displacement, W")
       plot!(xfine, wExact[i], label="exact", color=color[4], lw=1.5)
       
       plot!(twinx(),xfine, werr[i], ylims=(-0.02, 0.00),
             yaxis=("Error\n |w_exact - w_approx|"), color=color[9],
             y_guidefontcolor=color[9], legend=false)
             
  p2 = plot(xfine, uNet[i],color=color[1],
           lw=1.5, legend=:topright, ylims=(-1.2, 0),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="Stress, u = Wₓ")
      plot!(xfine, uExact[i], label="exact", color=color[4], lw=1.5)
       
      plot!(twinx(),xfine, uerr[i], ylims=(-0.02, 0.02),
            yaxis=("Error\n |u_exact - u_approx|"), color=color[9],
            y_guidefontcolor=color[9], legend=false)
            
  p3 = plot(xfine, vNet[i],color=color[1],
           lw=1.5, legend=:topright,ylims=(-0.0, 4.0),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="Velocity, v = Wₜ")
      plot!(xfine, vExact[i], label="exact", color=color[4], lw=1.5)
       
      plot!(twinx(),xfine, verr[i], ylims=(-0.1, 0.1),
            yaxis=("Error\n |v_exact - v_approx|"), color=color[9],
            y_guidefontcolor=color[9], legend=false)
        
  p = plot(p1, p2, p3, layout=grid(3,1), size=(1000, 750))     
       end

gif(anim, "rate_and_state(1D_nonlin)fps30.gif", fps = 15)
""")


'''
Main.eval("""
using Plots
pyplot()
theme(:dark)    # the most powerful theme
i = 200
color = theme_palette(:dark).colors.colors
	p1 = plot(xfine, wNet[i], size=(1000, 750),color=color[1],
	    lw=1.5, legend=:topright,ylims=(-0.0, 2.2),
	    label = "network", xaxis=(""), linestyle=:dashdot,
	    right_margin = 30Plots.mm, title="Displacement, W")
	plot!(xfine, wExact[i], label="exact", color=color[4], lw=1.5)

	plot!(twinx(),xfine, werr[i], ylims=(-0.02, -0.0),
	     yaxis=("Error\n |w_exact - w_approx|"), color=color[9],
	     y_guidefontcolor=color[9], legend=false)
	     
	p2 = plot(xfine, uNet[i], size=(1000, 750),color=color[1],
	   lw=1.5, legend=:topright, ylims=(-1.2, 0),
	   label = "network", xaxis=(""), linestyle=:dashdot,
	   right_margin = 30Plots.mm, title="Stress, u = Wₓ")
	plot!(xfine, uExact[i], label="exact", color=color[4], lw=1.5)

	plot!(twinx(),xfine, uerr[i], ylims=(0.0, 0.02),
	    yaxis=("Error\n |u_exact - u_approx|"), color=color[9],
	    y_guidefontcolor=color[9], legend=false)
	    
	p3 = plot(xfine, vNet[i], size=(1000, 750),color=color[1],
	   lw=1.5, legend=:topright,#ylims=(-0.0, 4.0),
	   label = "network", xaxis=(""), linestyle=:dashdot,
	   right_margin = 30Plots.mm, title="Velocity, v = Wₜ")
	plot!(xfine, vExact[i], label="exact", color=color[4], lw=1.5)

	plot!(twinx(),xfine, verr[i], ylims=(0.0, 0.07),
	    yaxis=("Error\n |v_exact - v_approx|"), color=color[9],
	    y_guidefontcolor=color[9], legend=false)

	p = plot(p1, p2, p3, layout=grid(3,1))  
display(p)   

""")
'''


'''
Main.eval("""
using Plots
pyplot()
theme(:dark)    # the most powerful theme
color = theme_palette(:dark).colors.colors
p=plot()
for i=1:15

	plot!(xfine, xfine .+i , size=(1000, 750),color=color[i],
	    lw=1.5)
	    
	    end
display(p)
""")
'''



