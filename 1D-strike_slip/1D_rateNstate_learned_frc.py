import torch 
import numpy as np
from tqdm import trange
from matplotlib import cm
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

num_iter = 10000
batch_size = 250
hidden_dim = 150
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
        
        self.sigmoid = torch.nn.Sigmoid()    
        
        
        self.n = 1.0  # adaptive weight scaling
        self.c1 = torch.nn.Parameter(torch.ones((1, hidden_dim), requires_grad=True) / self.n)  # adaptive weights a, b
        self.c2 = torch.nn.Parameter(torch.ones((1, hidden_dim), requires_grad=True) / self.n)
        self.a = torch.nn.Parameter(1*torch.ones((1, 1), requires_grad=True))
        self.b = torch.nn.Parameter(1*torch.ones((1, 1), requires_grad=True))
        self.Wa = torch.nn.Parameter(torch.rand((1,1), requires_grad=True))
        
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.01, betas=(0.9, 0.999), foreach=True)
        
    def forward(self, x):
        output =  self.fc1(x)
        output = self.sigmoid(self.n * self.c1 * output)
        #output = self.sigmoid(output)
      
        output = self.fc2(output)
        output = self.sigmoid(self.n * self.c2 * output)
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

def wexact_xt(z):
    x, t = z.split(1, dim=1)
    return -2*c*torch.tanh(x-c*t+1)*(1-torch.tanh(x-c*t+1)**2)
    
def wexact_tt(z):
    x, t = z.split(1, dim=1)
    return -2*(c**2)*torch.tanh(x-c*t+1)*(1-torch.tanh(x-c*t+1)**2)
    
def w_init(z):
    x, _ = z.split(1, dim=1)
    return 1 - torch.tanh(x + 1)
    
def v_init(z):
    x, _ = z.split(1, dim=1)
    return c * (1 - torch.tanh(x + 1)**2)

# exact function(and derivatives) for  ψ(t) derived from mms.   
def ψe(z):
    x, t = z.split(1, dim=1)
    return a * torch.log( (v0 / wexact_t(z)) * torch.sinh(-μ*wexact_x(z) / (a*σn)))

def ψe_t(z):
    return (-μ*wexact_xt(z)  / (torch.tanh(-μ*wexact_x(z) / (a * σn))*σn)) - (a  * wexact_tt(z) / (wexact_t(z)))
    
  
    
# Now, define NNets for solution u and state variable ψ
# establish necessary functions to be expressed in terms of out NNets
w_net = Feedforward(2,hidden_dim).to(device)  # displacement network
def w(z):
    x, t = z.split(1, dim=1)
    return w_init(z) + v_init(z)*t + w_net(z)*(t**2)

# state variable network
ψ_net = Feedforward(1,hidden_dim).to(device)
z0 = torch.tensor([[0.0, 0.0]])
ψ0 = ψe(z0).to(device)
def ψ(z):
    x, t = z.split(1, dim=1)
    return ψ0 + t*ψ_net(t)
    
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
    x, t = z.split(1, dim=1)
    return σn * w_net.a * torch.arcsinh((wexact_t(z)/v0) * torch.exp(ψ(z)/w_net.a))

# exact fault friction
def fe(z):
    x, t = z.split(1, dim=1) 
    return a * torch.arcsinh((wexact_t(z)/v0) * torch.exp(ψe(z) / a))
        
def fault_src(z):
    x, t = z.split(1, dim=1)
    #return (-μ - 2 * σn * c) * (1 - torch.tanh(x - c*t + 1)**2)
    return μ*wexact_x(z) - σn*fe(z)


# Determine source terms for the aging law
def Ge(z):
    return (b*v0/L) * torch.exp((f0 - ψe(z))/b - (2*wexact_t(z).abs() / v0))

def  Sψ(z):
    return ψe_t(z) - Ge(z)
    

def G(z):
    return (ψ_net.b*v0/L) * torch.exp((f0 - ψ(z))/ψ_net.b - (2*v(z).abs() / v0))  
    
def state_evoln(z):
    x, t = z.split(1, dim=1)
    grad_ψ, = torch.autograd.grad(ψ(z).sum(), z, create_graph=True)  # gradient of network w.r.t x,y
    dx, dt = grad_ψ.split(1,dim=1)  # gradient can be split into parts
    
    #return dt - G(z) - Sψ(z)
    return dt - Ge(z) - Sψ(z)  
    
          
def my_loss(z, z_x0, z_x02, z_x1, z_t):
    loss = wave_relation(z).sum() / z.shape[0]
    loss += (w(z_x02) - wexact(z_x02)).abs().square().sum() / z_x02.shape[0]
    loss += (μ*u(z_x0) - fault_str(z_x0) - fault_src(z_x0)).abs().square().sum() / z_x0.shape[0]
    loss += (Z*v(z_x1) + μ*u(z_x1)).abs().square().sum() / z_x1.shape[0]
    
    loss += state_evoln(z_t).abs().square().sum() / z_t.shape[0]
    #loss += (ψ_net(z_t) - ψe(z_t)).abs().square().sum() / z_t.shape[0]
    #loss += (w(z_t01) - wexact(z_t01)).abs().square().sum() / z_t01.shape[0]
    #loss += (v(z_t02) - wexact_t(z_t02)).abs().square().sum() / z_t02.shape[0]
    return loss
 
x0, x1 = [0.0, 1.0]
t0, t1 = [0.0, 2.0]
   
for i in trange(num_iter):  

    x = torch.rand((batch_size,1), requires_grad=True)
    t = Uniform(t0, t1).sample((batch_size,1)).requires_grad_(True)
    z = torch.cat([x, t], dim=1).to(device)

    x = x0 * torch.ones((batch_size,1), requires_grad=True)
    t = Uniform(t0, t1).sample((batch_size,1)).requires_grad_(True)
    #t = torch.tensor(np.random.choice(t_grid, batch_size), requires_grad=True).unsqueeze(1)
    z_x0 = torch.cat([x, t], dim=1).to(device)

    x = torch.rand((batch_size,1), requires_grad=True)
    t = Uniform(t0, t1).sample((batch_size,1)).requires_grad_(True)
    #t = torch.tensor(np.random.choice(t_grid, batch_size), requires_grad=True).unsqueeze(1)
    z_x02 = torch.cat([x, t], dim=1).to(device)
    
    x = x1 * torch.ones((batch_size,1), requires_grad=True)
    t = Uniform(t0, t1).sample((batch_size,1)).requires_grad_(True)
    #t = torch.tensor(np.random.choice(t_grid, batch_size), requires_grad=True).unsqueeze(1)
    z_x1 = torch.cat([x, t], dim=1).to(device)


    x = torch.zeros((batch_size,1), requires_grad=True)
    #x = torch.tensor(np.random.choice(x_grid, batch_size), requires_grad=True).unsqueeze(1)
    t = Uniform(t0, t1).sample((batch_size,1)).requires_grad_(True)
    z_t = torch.cat([x, t], dim=1).to(device)
    

    # Zero network gradients for every batch!
    w_net.opt.zero_grad()
    ψ_net.opt.zero_grad()
    #ψ_net.opt.zero_grad()
    #ψ.opt.zero_grad()

    # Make predictions for this batch
    #outputs = model(inputs)

    # Compute the loss and its gradients
    loss = my_loss(z, z_x0, z_x02, z_x1, z_t)
    loss.backward()

    # Adjust learning weights
    w_net.opt.step()
    ψ_net.opt.step()
    
    if w_net.a < 0.0001:
        w_net.a = torch.nn.Parameter(torch.tensor([[0.0001]], requires_grad=True).to(device))
        
    if ψ_net.b < 0.0001:
        ψ_net.b = torch.nn.Parameter(torch.tensor([[0.0001]], requires_grad=True).to(device))
        
    if i%1000 == 0:
        print(ψ_net.b)
        print(w_net.a)
        
print(ψ_net.b)
print(w_net.a)
  
t_grid = torch.arange(t0, 1.0, 0.005).requires_grad_().unsqueeze(1).to(device)
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

t_grid = torch.arange(t0, 1.0, 0.005).requires_grad_().unsqueeze(1).to(device)
#t_grid = np.arange(t0, t1, 0.001)
x_grid = 0*torch.arange(x0, x1, 0.005).requires_grad_().unsqueeze(1).to(device)
#t_grid = torch.zeros(x_grid.shape).requires_grad_().to(device)
z_grid2 = torch.cat([x_grid, t_grid], dim=1).to(device)

ψNet = ψ(z_grid2)
ψNet = ψNet.cpu().detach().numpy()

ψEx = ψe(z_grid2)
ψEx = ψEx.cpu().detach().numpy()
t = t_grid.cpu().detach().numpy()
x = x_grid.cpu().detach().numpy()

#plt.plot(t, p1, label="learned")
#plt.plot(t, p2, label="exact")
#plt.legend()
#plt.show()


from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

Main.xfine = x2_grid
Main.tfine = t_grid.cpu().detach().numpy()

Main.wNet = wNN
Main.wExact = wE
Main.werr = werr

Main.uNet = uNN
Main.uExact = uE
Main.uerr = uerr

Main.vNet = vNN
Main.vExact = vE
Main.verr = verr

Main.ψNet = ψNet
Main.ψEx = ψEx
Main.yfine = t
Main.xfine = x2_grid







dx = 0.05
dy = 0.05
dt = 0.005
#t_grid = torch.arange(0, 1+dt, dt).unsqueeze(1)
t_grid = torch.arange(0, 1+dy, dy).unsqueeze(1)
x_grid = torch.arange(0, 1+dx, dx).unsqueeze(1)

x0_grid = 0 *torch.arange(0, 1+dx, dx).unsqueeze(1)

U_NN = np.zeros((len(x_grid), len(t_grid)))
U = np.zeros((len(x_grid), len(t_grid)))
E = np.zeros((len(x_grid), len(t_grid)))


for j in range(len(t_grid)):
    for k in range(len(x_grid)):
        x = x_grid[k]
        t = t_grid[j]
            #t = t_grid[i]
        z = torch.cat([x, t], dim=0).to(device)
        z = z.unsqueeze(0)
            
        unn = w(z).item()
        ue = wexact(z).item()
        err = unn - ue
            
        U_NN[k, j] = unn
        U[k, j] = ue
        E[k, j] = err

#x0_grid = 0*torch.arange(x0, x1, 0.005).requires_grad_().unsqueeze(1).to(device)
#t_grid = torch.zeros(x_grid.shape).requires_grad_().to(device)
z_grid2 = torch.cat([x0_grid, t_grid], dim=1).to(device)

ψNet = ψ(z_grid2)
ψNet = ψNet.cpu().detach().numpy()

ψEx = ψe(z_grid2)
ψEx = ψEx.cpu().detach().numpy()

ψErr = ψNet - ψEx


from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.x = x_grid.cpu().detach().numpy()
Main.x = Main.vec(Main.x)
Main.y = t_grid.cpu().detach().numpy()    
Main.y = Main.vec(Main.y)
      
Main.t = t_grid.cpu().detach().numpy()

Main.X, Main.Y = np.meshgrid(x_grid.cpu().detach().numpy(), t_grid.cpu().detach().numpy())
Main.U_NN = U_NN
Main.U = U
Main.E = E   
Main.ψNet = ψNet
Main.ψEx = ψEx
Main.ψErr = ψErr



Main.eval("""
using Plots
pyplot()
theme(:default)         
stix = 20
plot(title="NNet \n")

	          
	p2 = plot(x, y, U_NN[:, :], title="Neural Net",colorbar = true, xlabel="x", ylabel="t",
           linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix, titlefontsize=25, leftmargin=10Plots.mm, yguidefontsize=stix, bottom_margin=40Plots.mm,
           colorbar_tickfontsize=stix)
	          
	
	          
	p4 = plot(x, y, U[:, :], title="Exact", xlabel="x", ylabel="t", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix)
	          
	          
	p6 = plot(x, y, E[:, :], title="Error", xlabel="x", ylabel="t", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix)
	          
	 
	p8 = plot(y, ψNet, size = (900, 700), dpi=300, lw=2.5, legend=:topright,
             label = "Neural Net", xaxis=("t"), right_margin = 30Plots.mm,ytickfontsize=stix,  xguidefontsize=stix,
             xtickfontsize=stix,yguidefontsize=stix, ylabel="", 
             title="How well Mᵧ Learns the Middle Integral N(b, y)-N(a, y)", titlefontsize=stix, 
             bottom_margin=60Plots.mm, legendfontsize=15, linestyle=:dashdot)
             
        plot!(y, ψEx, size = (900, 700), dpi=300, lw=2.5, legend=:topright,
             label = "Exact", xaxis=("t"), right_margin = 30Plots.mm,ytickfontsize=stix,  xguidefontsize=stix,
             xtickfontsize=stix,yguidefontsize=stix, ylabel="", 
             title="State Variable", titlefontsize=stix, 
             bottom_margin=60Plots.mm, legendfontsize=15)
         
        plot!(twinx(),y, ψErr, yaxis=("Error\n"), legend=false, yguidefontsize=stix, ytickfontsize=stix, lw=2.5, color="red", ytickfontcolor="red")
           
	#p = plot(p4, p2, p6, p8)   
	l = @layout [a b c; d]
        p = plot(p4,p2,p6,p8, layout=l, size = (2400, 1200), dpi=300)      
	#p = plot(p1,p2,p3, p4, p5, p6, layout=grid(3,2),
	#	             ylims = (0, 1.0), xlims = (0.0, 1.0),
	#	             size=(800, 1000))
savefig(p, "1D_fault_friction_results.eps")
""") 




stix = 20
plot(title="NNet \n", size = (900, 700))
#plot(x_grid, y_grid, Z[:,:], st=:contour)
plt1 = plot(xfine, yfine, Z1[:,:], title="Nₓ Learning f",colorbar = true, xlabel="x", ylabel="y",
           linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix, titlefontsize=25, leftmargin=10Plots.mm, yguidefontsize=stix, bottom_margin=40Plots.mm,
           colorbar_tickfontsize=stix)
#plt2 = plot(x_grid, y_grid, Z2[:,:], st=:surface)
plt2 = plot(xfine, yfine, Z3[:,:], title="Error", xlabel="x", ylabel="y", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix )
#plt4 = plot(x_grid, y_grid, Z4[:,:], st=:surface)
plt3 = plot(xfine, yfine, Z2[:,:], title="f(x, y) = cos(x²-y²)", xlabel="x", ylabel="y", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix)
#plt6 = plot(x_grid, y_grid, Z6[:,:], st=:surface)



plot(yfine, Z4.(yfine),#size = (900, 700), #dpi=300,
    lw=2.5, legend=:topleft, color=colors[3],
    label = "Mᵧ(y)", xaxis=("y"),
    right_margin = 30Plots.mm,ytickfontsize=stix,  xguidefontsize=stix, xtickfontsize=stix,yguidefontsize=stix,
    ylabel="", title="How well Mᵧ Learns the Middle Integral N(b, y)-N(a, y)", titlefontsize=stix, bottom_margin=60Plots.mm, legendfontsize=15,
    linestyle=:dashdot)

plot!(yfine, Z5.(yfine), label="N(b, y) - N(a, y)", color=colors[1],lw=2.5  )#, linestyle=:dashdot)

plt4 = plot!(twinx(),yfine, Z6.(yfine), yaxis=("Error\n |Mᵧ - N(x, y)| "), color=colors[9], ytickfontcolor=colors[9],
y_guidefontcolor=colors[9], legend=false, yguidefontsize=stix, ytickfontsize=stix, lw=2.5)#, inset = (1,bbox(0.05, 0.05, 0.5, 0.25, :bottom, :right)), subplot=2 )

plt_hold = plot(legend=false,grid=false, showaxis = false)
l = @layout [a b c; d]
p = plot(plt3,plt1,plt2,plt4, layout=l, size = (2400, 1200), dpi=300)




Main.Z1 = Z1
Main.Z2 = Z2
Main.Z3 = Z3

Main.eval("""
using Plots
stix = 20
plt1 = plot(xfine, yfine, Z1[:,:], title="Nₓ Learning f",colorbar = true, xlabel="x", ylabel="y",
           linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix, titlefontsize=25, leftmargin=10Plots.mm, yguidefontsize=stix, bottom_margin=40Plots.mm,
           colorbar_tickfontsize=stix)
display(plt1)
""")
Main.eval("""

#Z1 = zeros(length(xfine), length(tfine))
#Z2 = zeros(length(xfine), length(tfine))
#Z3 = zeros(length(xfine), length(tfine))


#for i = 1:length(xfine)
#    for j = 1:length(tfine)
#        Z1[i, j] = Nₓ(xfine[i], yfine[j], Σ[1:9]...)
#        Z2[i, j] = u(xfine[i], yfine[j])
#        Z3[i, j] = abs.(Nₓ(xfine[i], yfine[j], Σ[1:9]...) - u(xfine[i], yfine[j]))


#    end
#end


Z4(y) = N(b, y, Σ[1:8]...) - N(a,y, Σ[1:8]...)
Z5(y) = Mᵧ(y, Σ[9:15]...)
Z6(y) = abs.(Mᵧ(y, Σ[9:15]...) - (N(b, y, Σ[1:8]...) - N(a, y, Σ[1:8]...)))

pyplot()
theme(:lime)
colors = theme_palette(:lime).colors.colors

stix = 20
plot(title="NNet \n", size = (900, 700))
#plot(x_grid, y_grid, Z[:,:], st=:contour)
plt1 = plot(xfine, yfine, Z1[:,:], title="Nₓ Learning f",colorbar = true, xlabel="x", ylabel="y",
           linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix, titlefontsize=25, leftmargin=10Plots.mm, yguidefontsize=stix, bottom_margin=40Plots.mm,
           colorbar_tickfontsize=stix)
#plt2 = plot(x_grid, y_grid, Z2[:,:], st=:surface)
plt2 = plot(xfine, yfine, Z3[:,:], title="Error", xlabel="x", ylabel="y", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix )
#plt4 = plot(x_grid, y_grid, Z4[:,:], st=:surface)
plt3 = plot(xfine, yfine, Z2[:,:], title="f(x, y) = cos(x²-y²)", xlabel="x", ylabel="y", colorbar=true, linetype=:contourf, xguidefontsize=stix, xtickfontsize=stix, ytickfontsize=stix,  titlefontsize=stix,
colorbar_tickfontsize=stix,
yguidefontsize=stix)
#plt6 = plot(x_grid, y_grid, Z6[:,:], st=:surface)



plot(yfine, Z4.(yfine),#size = (900, 700), #dpi=300,
    lw=2.5, legend=:topleft, color=colors[3],
    label = "Mᵧ(y)", xaxis=("y"),
    right_margin = 30Plots.mm,ytickfontsize=stix,  xguidefontsize=stix, xtickfontsize=stix,yguidefontsize=stix,
    ylabel="", title="How well Mᵧ Learns the Middle Integral N(b, y)-N(a, y)", titlefontsize=stix, bottom_margin=60Plots.mm, legendfontsize=15,
    linestyle=:dashdot)

plot!(yfine, Z5.(yfine), label="N(b, y) - N(a, y)", color=colors[1],lw=2.5  )#, linestyle=:dashdot)

plt4 = plot!(twinx(),yfine, Z6.(yfine), yaxis=("Error\n |Mᵧ - N(x, y)| "), color=colors[9], ytickfontcolor=colors[9],
y_guidefontcolor=colors[9], legend=false, yguidefontsize=stix, ytickfontsize=stix, lw=2.5)#, inset = (1,bbox(0.05, 0.05, 0.5, 0.25, :bottom, :right)), subplot=2 )

plt_hold = plot(legend=false,grid=false, showaxis = false)
l = @layout [a b c; d]
p = plot(plt3,plt1,plt2,plt4, layout=l, size = (2400, 1200), dpi=300)



savefig(p, "2D_integral.png")
""")
'''
plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(15,12))
fig.tight_layout()
ax1.set_title('t=0.0')
ax2.set_title('t=0.5')
ax3.set_title('t=1.0')
ax4.set_title('State Variable')

axe1 = ax1.twinx()
axe2 = ax2.twinx()
axe3 = ax3.twinx()

ax1.plot(x2_grid, wE[0], label="Exact", color=colors[1], lw=3)
ax1.plot(x2_grid, wNN[0], label="Neural Net", linestyle=':', color='black', lw=3)

ax1.set_xlabel("x")
ax1.set_ylabel("Displacement")
ax1.set_ylim(-0.1, 0.5)

axe1.plot(x2_grid, werr[0], label="Error", lw=2)
axe1.set_ylabel("Error", color=colors[0])

ax2.plot(x2_grid, wE[100], label="Exact", color=colors[1], lw=3)
ax2.plot(x2_grid, wNN[100], label="Neural Net", linestyle=':', color='black', lw=3)

ax2.set_xlabel("x")
ax2.set_ylabel("Displacement")
ax2.set_ylim(-0.1, 2.5)

axe2.plot(x2_grid, werr[100], label="Error", lw=2)
axe2.set_ylabel("Error", color=colors[0])


ax1.legend(bbox_to_anchor=(1.21, 1.05), loc='upper right', prop={'size': 16}, markerscale=5)

ax3.plot(x2_grid, wE[-1], label="Exact", color=colors[1], lw=3)
ax3.plot(x2_grid, wNN[-1], label="Neural Net", linestyle=':', color='black', lw=3)

ax3.set_xlabel("x")
ax3.set_ylabel("Displacement")
ax3.set_ylim(-0.1, 2.5)

axe3.plot(x2_grid, werr[-1], label="Error", lw=2)
axe3.set_ylabel("Error", color=colors[0])

ax4.plot(t, ψEx, label="Exact", color=colors[1], lw=3)
ax4.plot(t, ψNet, label="Neural Net", linestyle=':', color='black', lw=3)

ax4.set_xlabel("t")
ax4.set_ylabel("State")
#ax4.set_ylim(0, 3.0)

plt.subplots_adjust(hspace=0.5)
fig.savefig('./NNet_comparisons.png',bbox_inches="tight", dpi=300)



plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set_title('State Variable')


#axe = ax.twinx()

ax.plot(t, ψEx, label="Exact", color=colors[1])
ax.plot(t, ψNet, label="Neural Net", linestyle=':', color='black')

ax.set_xlabel("t")
ax.set_ylabel("State")
#ax.set_ylim(0, 3.0)

#axe.plot(x2_grid, werr[1], label="Error")
#axe.set_ylabel("Error", color=colors[0])
ax.legend(bbox_to_anchor=(0.95, 1), loc='upper right')
fig.savefig('./NNet_State_var.eps',bbox_inches="tight", dpi=300)

'''

'''
# visualize solution
Main.eval("""
using Plots

theme(:ggplot2)    # the most powerful theme
color = theme_palette(:ggplot2).colors.colors

anim = @animate for i=1:length(tfine)

  p1 = plot(xfine, wNet[i], size=(1000, 750),color=color[1],
            lw=1.5, legend=:topright,ylims=(-0.0, 2.1),
            label = "network", xaxis=(""), linestyle=:dashdot,
            right_margin = 30Plots.mm, title="Displacement, W")
       plot!(xfine, wExact[i], label="exact", color=color[4], lw=1.5)
       
       plot!(twinx(),xfine, werr[i], ylims=(-0.1, 0.1),
             yaxis=("Error\n |w_exact - w_approx|"), color=color[9],
             y_guidefontcolor=color[9], legend=false)
             
  p2 = plot(xfine, uNet[i], size=(1000, 750),color=color[1],
           lw=1.5, legend=:topright,ylims=(-1.2, 0),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="Stress, u = Wₓ")
      plot!(xfine, uExact[i], label="exact", color=color[4], lw=1.5)
       
      plot!(twinx(),xfine, uerr[i], ylims=(-0.1, 0.1),
            yaxis=("Error\n |u_exact - u_approx|"), color=color[9],
            y_guidefontcolor=color[9], legend=false)
            
  p3 = plot(xfine, vNet[i], size=(1000, 750),color=color[1],
           lw=1.5, legend=:topright,ylims=(-0.0, 3.6),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="Velocity, v = Wₜ")
      plot!(xfine, vExact[i], label="exact", color=color[4], lw=1.5)
       
      plot!(twinx(),xfine, verr[i], ylims=(-0.2, 0.3),
            yaxis=("Error\n |v_exact - v_approx|"), color=color[9],
            y_guidefontcolor=color[9], legend=false)
            
  p4 = plot(t, ψNet, size=(1000, 750),color=color[1],
           lw=1.5, legend=:topright,#ylims=(-0.0, 1.2),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="State variable, v = Wₜ")
      plot!(t, ψEx, label="exact", color=color[4], lw=1.5)
      
            
  p = plot(p1, p2, p3,p4,  layout=grid(4,1))     
       end

gif(anim, "rate_and_state(1D)fps30.gif", fps = 30)
""")




Main.eval("""
using Plots
#pyplot()
theme(:default)    # the most powerful theme
color = theme_palette(:default).colors.colors
p=plot()
for i=1:15

	plot!(xfine, xfine .+i , size=(1000, 750),color=color[i],
	    lw=1.5)
	    
	    end
display(p)
""")



Main.eval("""
using Plots

theme(:default)    # the most powerful theme
color = theme_palette(:default).colors.colors



  p1 = plot(xfine, wNet[1], size=(1000, 750),color=color[1],
            lw=1.5, legend=:topright,ylims=(-0.0, 2.1),
            label = "network", xaxis=(""), linestyle=:dashdot,
            right_margin = 30Plots.mm, title="Displacement, W")
       plot!(xfine, wExact[1], label="exact", color=color[4], lw=1.5)
       
       plot!(twinx(),xfine, werr[1], ylims=(-0.1, 0.1),
             yaxis=("Error\n |w_exact - w_approx|"), color=color[9],
             y_guidefontcolor=color[9], legend=false)
             
  p2 = plot(xfine, wNet[1], size=(1000, 750),color=color[1],
           lw=1.5, legend=:topright,ylims=(-1.2, 0),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="Stress, u = Wₓ")
      plot!(xfine, wExact[1], label="exact", color=color[4], lw=1.5)
       
      plot!(twinx(),xfine, werr[1], ylims=(-0.1, 0.1),
            yaxis=("Error\n |u_exact - u_approx|"), color=color[9],
            y_guidefontcolor=color[9], legend=false)
            
  #p3 = plot(xfine, vNet[i], size=(1000, 750),color=color[1],
   #        lw=1.5, legend=:topright,ylims=(-0.0, 3.6),
    #       label = "network", xaxis=(""), linestyle=:dashdot,
     #      right_margin = 30Plots.mm, title="Velocity, v = Wₜ")
      #plot!(xfine, vExact[i], label="exact", color=color[4], lw=1.5)
       
      #plot!(twinx(),xfine, verr[i], ylims=(-0.2, 0.3),
       #     yaxis=("Error\n |v_exact - v_approx|"), color=color[9],
        #    y_guidefontcolor=color[9], legend=false)
            
  p4 = plot(t, ψNet, size=(1000, 750),color=color[1],
           lw=1.5, legend=:topright,#ylims=(-0.0, 1.2),
           label = "network", xaxis=(""), linestyle=:dashdot,
           right_margin = 30Plots.mm, title="State variable, v = Wₜ")
      plot!(t, ψEx, label="exact", color=color[4], lw=1.5)
      
            
  p = plot(p1, p2, p3,p4,  layout=grid(4,1)) 
  display(p)    

""")

'''


