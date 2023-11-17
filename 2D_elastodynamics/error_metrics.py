from FeedForward import *
from geometry_2D import *
from NNet_viz import *
from IBVP import *
import torch
from tqdm import trange
import StrikeSlipMMS as MMS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

N_layers = np.array([3, 128, 128, 128, 1])
a_layers = np.array([1, 128, 128, 128, 128, 128, 1])
activations = [nn.Tanh(), nn.Tanh(), nn.Tanh()]
a_activations = [nn.Tanh(), nn.ReLU(), nn.Tanh(),nn.ReLU(), nn.Tanh()]

N = Feedforward(N_layers, activations, adaptive_weight='neuron').to(device)
a_net = Feedforward(a_layers, a_activations, adaptive_weight='neuron').to(device)
#N.a = torch.nn.Parameter(torch.rand(1)).to(device)

params =  params = list(N.parameters()) +list(a_net.parameters())
opt = torch.optim.LBFGS(params, lr=0.1, max_iter=250,
                        max_eval=None, tolerance_grad=1e-05,
                        tolerance_change=1e-09, history_size=100,
                        line_search_fn='strong_wolfe')
                        
opt2 = torch.optim.Adam(N.parameters(), lr=0.01, betas=(0.9, 0.999), foreach=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10)


toplft, botlft, botrgt, toprgt = [
    torch.tensor([[x, y]]) for (x, y) in [
            (0.0, 25.0),(0.0, 0.0),(25.0, 0.0), (25.0, 25.0)]]




#def uexact(z):
#    x, y, t = z.split(1, dim=1)
#    return torch.sin(2*torch.pi*(x - t)) + torch.sin(2*torch.pi*(y - t))


#def uexact_t(z):
#    x, y, t = z.split(1, dim=1)
#    return -2*torch.pi * torch.cos(2*torch.pi*x) -2 * torch.pi * torch.cos(2*torch.pi*y)

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

mms = MMS.StrikeSlipMMS(a=a, b=b)
uexact = mms.uexact
uexact_t = mms.uexact_t
sexact = mms.ψe

def bot_src(z):
    return Z*mms.uexact_t(z)

def friction_coeff(z):
    x, y, t = z.split(1, dim=1)
    
    
    y1 = torch.where(y < 12, -0.005, 0.0)
    y2 = torch.where(y > 17, 0.015, 0.0)
    y3 = torch.where((y >= 12) & (y <= 17), (y - 12) * ((0.015+0.005)/5) - 0.005, 0.0)
    out = y1+y2+y3
        
    return out
    
def fault_source(z):
    eps = torch.tensor([10e-12])
    fault_str = f0 + friction_coeff(z)*torch.log((mms.uexact_t(z) / (2*v0)) + eps)
    fault = (-μ * mms.uexact_x(z)) - σn*fault_str

    return fault
    
def fault_check(x):
    z = x.clone().detach().requires_grad_(True)
    z = z.to(device)
    
    out = N(z)
    grad, = torch.autograd.grad(out.sum(), z, create_graph=True)
    dx, dy, dt = grad.split(1, dim=1)
    
    fault_str = f0 + (N.a)*torch.log(dt / (2*v0))
    fault = (-μ * dx) - σn*fault_str

    return fault

def func_learn(z):
    _, depth, _ = z.split(1, dim=1)
    fault_str = f0 + a_net(depth)*torch.log(mms.uexact_t(z) / (2*v0))
    fault = (-μ * mms.uexact_x(z)) - σn*fault_str

    return fault

def top_source(z):
    return mms.μ*mms.uexact_y(z) + mms.Z * mms.uexact_t(z)
       
BC = {}
BC['lft'] = BoundaryCurve(['line', toplft, botlft])
BC['bot'] = BoundaryCurve(['line', botlft, botrgt])
BC['rgt'] = BoundaryCurve(['line', botrgt, toprgt])
BC['top'] = BoundaryCurve(['line', toprgt, toplft])

IB = IBVP(SpaceTimeDomain(BC))

#for key in BC:
#    IB.set_bc(NNet=N, data_function=uexact, bc='dirichlet', label=key)
'''
IB.BC['lft'] = RateFault(IB.domain.bdry['lft'], NNet=N, data_function=fault_source)
#IB.BC['lft'] = BoundaryCondition(IB.domain.bdry['lft'], NNet=N, data_function=uexact)
IB.BC['bot'] = Characteristic(IB.domain.bdry['bot'], NNet=N, data_function=bot_src)
IB.BC['rgt'] = Characteristic(IB.domain.bdry['rgt'], NNet=N, data_function=None)
IB.BC['top'] = BoundaryCondition(IB.domain.bdry['top'], NNet=N, data_function=None, bc='neumann')
'''

IB.BC['lft'] = RateFault(IB.domain.bdry['lft'], NNet=N, fric_net=a_net, data_function=fault_source, fixed=False)
#IB.BC['lft'] = BoundaryCondition(IB.domain.bdry['lft'], NNet=N, data_function=uexact)
IB.BC['bot'] = BoundaryCondition(IB.domain.bdry['bot'], NNet=N, data_function=None, bc='neumann')
IB.BC['rgt'] = Characteristic(IB.domain.bdry['rgt'], NNet=N, data_function=None)
IB.BC['top'] = Characteristic(IB.domain.bdry['top'], NNet=N, data_function=top_source)

IB.set_ic(NNet=N, data_function=uexact, bc='primal', label='u0')
IB.IC['u0'].t_final = 0.0

IB.set_ic(NNet=N, data_function=uexact_t, bc='velocity', label='v0')
IB.IC['v0'].t_final = 0.0

N.load_state_dict(torch.load('./model_N2.pth'))

##################################################
import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

get_gpu_memory()

###################################################
#sample_size = 1000
#Nx = 80
#Ny = 80
#grid = IB.domain.spatial_grid(Nt=Nt, Ns=Ns).to(device)
#Δx = grid[1,0] - grid[0,0]
#Δy = grid[Nt+1,1] - grid[0,1]
#Lx, Ly = grid[-1,:]
#Nx = int(Lx/Δx)
#Ny = int(Ly/Δy)

#E = torch.tensor([]).to(device)

#t_dist = torch.rand(sample_size,1).to(device)

#L2_xy_dist = []

def Spatial_L2(grid, sample_size, t_grid=None, dx=None, dy=None):
    Δx = grid[1,0] - grid[0,0]
    Δy = grid[Nt+1,1] - grid[0,1]
    Lx, Ly = grid[-1,:]
    Nx = round((Lx/Δx).item())
    Ny = round((Ly/Δy).item())
    
    if t_grid is None:
        t_dist = torch.rand(sample_size,1).to(device)
    else:
        t_dist = t_grid
        
    L2_xy_dist = []
    for time in t_dist:
        E = torch.tensor([]).to(device)
    #t = torch.tile(time, (grid.shape[0],1))
    #grid = torch.cat([grid, t], dim=1)
        for j in range(Ny+1):
            sub_grid = grid[j*(Nx+1):(j+1)*(Nx+1)].to(device)
            t = torch.tile(time, (sub_grid.shape[0],1))
            sub_grid = torch.cat([sub_grid, t], dim=1)
        
            err = (N(sub_grid) - mms.uexact(sub_grid)).square()    
            err = IB.domain.quadrature_1D(err, Δx)
    
            E = torch.cat([E, err], dim=0)

        E = E.view(Ny+1, 1)
        E = IB.domain.quadrature_1D(E, Δy)
        E = torch.sqrt(E)
        L2_xy_dist.append(E.cpu().detach())
    return L2_xy_dist

#Nt = 80
#dt = 1.0 / Nt
#t_grid = torch.arange(0.0, 1.0+dt, dt).unsqueeze(1).to(device)

def temporal_L2(t_grid, sample_size, xy_grid=None):#z=None, Nt=10, ti=0.0, tf=1.0):
        #T0, T1 = [ti, tf]
        Δt = t_grid[1] - t_grid[0]
        
        #t_grid = torch.arange(T0, T1+Δt, Δt)
        #t_grid = t_grid.unsqueeze(1).to(device)
        
        if xy_grid is None:
            z = IB.domain.get_point(n_samples=sample_size).to(device)
         
        else:
            z = xy_grid    
        time_extension = [torch.cat([torch.tile(x, (len(t_grid),1)), t_grid], dim=1) for x in z[:]]
        
        ERR = torch.tensor([])
        
        for time_block in time_extension:
            err = (N(time_block) - mms.uexact(time_block)).square()
            err = IB.domain.quadrature_1D(err, Δt)
            err = torch.sqrt(err).cpu().detach()  
            ERR = torch.cat([ERR, err.view(1,1)], dim=1)
            
        #pt_tensor = torch.tile(z, (len(t_grid), 1)) 
        #pt_tensor = torch.cat([pt_tensor, t_grid], dim=1)
        
        #err = (f1(pt_tensor) - f2(pt_tensor)).square()
        #err = self.quadrature_1D(err, Δt)
        #err = torch.sqrt(err)
        
        #ERR = ERR.t()
        #z = torch.cat([z, ERR], dim=1)
        return ERR.t()

def xyt_L2(xy_grid, t_grid, Nx, Ny, Nt):
    Δx = xy_grid[1,0] - xy_grid[0,0]
    Δy = xy_grid[Nt+1,1] - xy_grid[0,1]
    Δt = t_grid[1] - t_grid[0]
    
    Lx, Ly = xy_grid[-1,:]
    Lt = t_grid[-1]
    
    Nx = round((Lx/Δx).item())
    Ny = round((Ly/Δy).item())
    Nt = round((Lt/Δt).item())
    
    quad_xy = torch.tensor([])
    for time in t_grid:
        E = torch.tensor([]).to(device)
    #t = torch.tile(time, (grid.shape[0],1))
    #grid = torch.cat([grid, t], dim=1)
        for j in range(Ny+1):
            sub_grid = xy_grid[j*(Nx+1):(j+1)*(Nx+1)].to(device)
            t = torch.tile(time, (sub_grid.shape[0],1))
            sub_grid = torch.cat([sub_grid, t], dim=1)
        
            err = (N(sub_grid) - mms.uexact(sub_grid)).square()    
            err = IB.domain.quadrature_1D(err, Δx)
    
            E = torch.cat([E, err], dim=0)

        E = E.view(Ny+1, 1)
        E = IB.domain.quadrature_1D(E, Δy).cpu().detach()
        
        quad_xy = torch.cat([quad_xy, E], dim=0)
        
    quad_xy = quad_xy.view(quad_xy.shape[0],1)    
    quad_xyt = IB.domain.quadrature_1D(quad_xy.to(device), Δt)
    quad_xyt = quad_xyt.sqrt()
    return quad_xyt

sample_size = 1000
Nx = 80
Ny = 80
Nt = 80
grid = IB.domain.spatial_grid(Nt=Nx, Ns=Ny).to(device)
dt = 1.0 / Nt
t_grid = torch.arange(0.0, 1.0+dt, dt).unsqueeze(1).to(device)

grid_spacing = [20, 40, 80, 160, 320]
sample_size = 1000

spatial = {}
spatial['mu'] = []
spatial['sig'] = []

temporal = {}
temporal['mu'] = []
temporal['sig'] = []

total = []
for i in trange(len(grid_spacing)):
    h = grid_spacing[i]
    Nx = h
    Ny = h
    Nt = h
    
    grid = IB.domain.spatial_grid(Nt=Nx, Ns=Ny).to(device)
    dt = 1.0 / Nt
    t_grid = torch.arange(0.0, 1.0+dt, dt).unsqueeze(1).to(device)
    
    xy_dist = Spatial_L2(grid, sample_size)
    t_dist = temporal_L2(t_grid, sample_size)
    total.append(xyt_L2(grid, t_grid, Nx, Ny, Nt).item())
    
    spatial['mu'].append(torch.mean(torch.tensor(xy_dist)).item())
    spatial['sig'].append(torch.var(torch.tensor(xy_dist)).item())

    temporal['mu'].append(torch.mean(t_dist).item())
    temporal['sig'].append(torch.var(t_dist).item())

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, figsize=(5.5, 5.5))
ax.set_xlabel('Grid Size')   
ax.set_ylabel('L2-Error')  

ax.plot(grid_spacing, spatial['mu'], marker='|', markersize=12, c=colors[4], label="Spatial Error")
#ax.plot(grid_spacing, spatial['sig'], marker='v', markersize=10, c=colors[4])

ax.plot(grid_spacing, temporal['mu'], marker='|', markersize=12, c=colors[1], label="Temporal Error")
#ax.plot(grid_spacing, temporal['sig'], marker='v', markersize=10, c=colors[1])

ax.plot(grid_spacing, total, marker='|', markersize=12, c=colors[0], label="Total Error")

plt.yscale("log")
plt.xscale("log")
plt.legend()
fig.savefig("./plots/L2_Error_Plot_InverseProblem_nodata_2.eps" , dpi=300, format='eps')

for i in range(len(colors)):
    plt.plot([0,1], [i,i], label="{}".format(i))
    
    
'''
xy_dist = Spatial_L2(grid, sample_size)
t_dist = temporal_L2(t_grid, sample_size)
total = xyt_L2(grid, t_grid, Nx, Ny, Nt)

spatial_mean = torch.mean(torch.tensor(xy_dist)).item()
spatial_var = torch.var(torch.tensor(xy_dist)).item()

temporal_mean = torch.mean(t_dist).item()
temporal_var = torch.var(t_dist).item()

print("\mu$&{:.5e}&{:.5e}".format(spatial_mean, temporal_mean))
#\cline{2-4}
print("\sigma$&{:.5e}&{:.5e}".format(spatial_var, temporal_var))

print("{:.5e}".format(total.item()))
'''

Nx = 80
Ny = 80
Nt = 80
grid = IB.domain.spatial_grid(Nt=Nx, Ns=Ny).to(device)
dt = 1.0 / Nt
t_grid = torch.arange(0.0, 1.0+dt, dt).unsqueeze(1).to(device)

def interpolate_func(grid, f, time):
        t = torch.tile(time, (grid.shape[0], 1)).to(device)
        
        xyt_grid = torch.cat([grid, t], dim=1)
        u = f(xyt_grid)
        xyu_grid = torch.cat([grid, u], dim=1)
        return xyu_grid

t = torch.tensor([[1.0]]).to(device)        
uN = interpolate_func(grid, N, t)
x, y, uN = IB.domain.get_mesh(uN, Nt=Nx, Ns=Ny)

uE = interpolate_func(grid, mms.uexact, t)   
_, _, uE = IB.domain.get_mesh(uE, Nt=Nx, Ns=Ny)     

L2_t = temporal_L2(t_grid, 80, xy_grid=grid)
_, _, L2_t = IB.domain.get_mesh(L2_t, Nt=80, Ns=80)

X = x.cpu().detach().numpy()
Y = y.cpu().detach().numpy()
ZN = uN.cpu().detach().numpy()
ZE = uE.cpu().detach().numpy()

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_ylim(25,0)
ax1.set_ylabel('Depth (km)')    

ax1.xaxis.tick_top()
ax1.set_xlabel('Distance (km)')    
ax1.xaxis.set_label_position('top') 

cp1 = ax1.contourf(X, Y, L2_t)

ax2.set_ylim(25,0)
ax2.set_ylabel('Depth (km)')    

ax2.xaxis.tick_top()
ax2.set_xlabel('Distance (km)')    
ax2.xaxis.set_label_position('top') 

cp2 = ax2.contourf(X, Y, ZE)

plt.colorbar(cp1)
plt.grid(visible=True)
#plt.savefig("./temporal_L2_across_domain.eps" , dpi=300, format='eps')


fig, ax = plt.subplots(1, figsize=(5, 5))#figsize=(800,500))
#tit = fig.suptitle('Learning Depth-Dependent Friction Parameter\n',
                  # fontsize=14)
#fig.tight_layout(h_pad=10)
#tit.set_y(1.0)
#cmap = plt.get_cmap('plasma_r')
#plt.set_cmap(cmap)
#color_index = np.arange(0, 256, 85).tolist()

ax.set_ylabel("Spatial L2 Error")
ax.set_xlabel("t")
#ax.set_xlim(-0.025, 0.025)
#ax.plot(t, l2,linewidth=2.0)
ax.plot(t, sN,linewidth=2.0, label='NNet Approx.')
ax.plot(t, sE,linewidth=2.0, label='Exact Solution')
fig.savefig("./plots/State_variable_fig" , dpi=300, format='eps')
#fig.subplots_adjust(bottom=0.09, left=0.125)
#u0 = training_progress[0].cpu().detach().numpy()


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

m1 = np.random.rand(3, 3)
m2 = np.arange(0, 3*3, 1).reshape((3, 3))


cmap = plt.colormaps["viridis"]#.with_extremes(under="magenta", over="yellow")
fig = plt.figure(figsize=(12, 6))
#fig, ax1 = plt.subplots(figsize=(6, 6))
fig.tight_layout(pad=5.0)

ax1 = fig.add_subplot(121)
ax1.set_title("NNet Approx.")
ax1.set_ylim(25,0)
ax1.set_ylabel('Depth (km)')    

ax1.xaxis.tick_top()
ax1.set_xlabel('Distance (km)')    
ax1.xaxis.set_label_position('top') 

im1 = ax1.contourf(X, Y, ZN, cmap=cmap, levels=20, vmin=0, vmax=1)

c_tix = np.arange(0, 1+0.1, 0.1).tolist()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical', ticks=c_tix)
#fig.savefig("./plots/L2_time_error_at_each_pt_in_space.eps" , dpi=300, format='eps')

ax2 = fig.add_subplot(122)
ax2.set_title("Exact Solution")
ax2.set_ylim(25,0)
ax2.set_ylabel('Depth (km)')    

ax2.xaxis.tick_top()
ax2.set_xlabel('Distance (km)')    
ax2.xaxis.set_label_position('top') 
im2 = ax2.contourf(X, Y, ZE, cmap=cmap, levels=20, vmin=0, vmax=1)

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical', ticks=c_tix)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

fig.savefig("./plots/Displacements_hardBC_learned_ab_nodata.eps" , dpi=300, format='eps')


x_grid = grid[:,0].cpu().detach().numpy()
y_grid = grid[:,0].cpu().detach().numpy()
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.x = X
#Main.x = Main.vec(Main.x)
Main.y = Y    
#Main.y = Main.vec(Main.y)
      

#Main.X, Main.Y = np.meshgrid(x_grid.cpu().detach().numpy(), y_grid.cpu().detach().numpy())
Main.uN = ZN
Main.uE = ZE
  
Main.eval("""
using Plots
pyplot()
theme(:default)     
	p1 = plot(x, y, uN, st=:contourf, levels=15, title="NNet Approx.", yflip=true,
	xlabel="Distance (km)", ylabel="Depth (km)")
	
	p2 = plot(x, y, uE, st=:contourf, levels=15,title="Exact Solution", yflip=true, colorbar=true,
	          xlabel="Distance (km)", ylabel="Depth (km)")          
	
	          
	          
	p = plot(p1, p2, size=(1400, 1000), layout=grid(1,2))
	display(p)
""") 

x0, x1 = [0.0, 25.0]
y0, y1 = [0.0, 25.0]
δx = 0.5
δy = 0.5

y_grid = torch.arange(y0, y1+δy, δy).unsqueeze(1)
x_grid = torch.arange(x0, x1+δx, δx).unsqueeze(1)

U_NN = np.zeros((len(x_grid), len(y_grid)))
U = np.zeros((len(x_grid), len(y_grid)))
#psi = np.zeros((len(x_grid), len(y_grid), len(t_grid)))
#psi_ex = np.zeros((len(x_grid), len(y_grid), len(t_grid)))



for j in range(len(y_grid)):
    for k in range(len(x_grid)):
        
        x = x_grid[k]
        y = y_grid[j]
        t = torch.tensor([1.0])
        z = torch.cat([x, y, t], dim=0).to(device)
        z = z.unsqueeze(0)

            
        unn = N(z).item()
        ue = mms.uexact(z).item()
         
        U_NN[j, k] = unn
        U[j, k] = ue

            
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.x = x_grid.cpu().detach().numpy()
Main.x = Main.vec(Main.x)
Main.y = y_grid.cpu().detach().numpy()    
Main.y = Main.vec(Main.y)
      


Main.U_NN = U_NN
Main.U = U
        


Main.eval("""
using Plots
using Measures  
theme(:default) 
p1 = contourf(x, y, U_NN[:,:], xlabel="Distance (km)",levels=20, ylabel="Depth (km)", title="NNet Approx.", yflip=true,
    guidefontsize=18, tickfontsize=16, titlefontsize=20, clims=(0, 1.0))
p2 = contourf(x, y, U[:,:], xlabel="Distance (km)", levels=20, ylabel="Depth (km)", title="Exact Soln.", yflip=true,
    guidefontsize=18, tickfontsize=16, titlefontsize=20, clims=(0, 1.0))

p = plot(p1, p2, layout=grid(1,2), size=(2000,800), dpi=300, margin=15mm)
savefig("./Displacement_plot_second.png")
""")


