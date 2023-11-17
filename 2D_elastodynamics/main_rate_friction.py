from FeedForward import *
from geometry_2D import *
from NNet_viz import *
from IBVP import *
import torch
from tqdm import trange
import StrikeSlipMMS as MMS



        
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

N_layers = np.array([3, 3, 3, 3, 1])
a_layers = np.array([1, 3, 3, 3, 1])
activations = [nn.SiLU(), nn.SiLU(), nn.SiLU()]
a_activations = [nn.SiLU(), nn.SiLU(), nn.SiLU()]#, nn.Tanh(), nn.Tanh()]

#N_net = FeedforwardAdapt(N_layers, activations, adaptive_weight='neuron').to(device)
#a_net = FeedforwardAdapt(a_layers, a_activations, adaptive_weight='neuron').to(device)

N_net = Feedforward(N_layers, activations).to(device)
a_net = Feedforward(a_layers, a_activations).to(device)
#a_net = FeedforwardAdapt(a_layers, a_activations, adaptive_weight='neuron').to(device)
#N.a = torch.nn.Parameter(torch.rand(1)).to(device)



def N(z):
    x, y, t = z.split(1, dim=1)  
    t0 = torch.zeros(x.shape).to(device)
    
    z0 = torch.cat([x, y, t0], dim=1)
    
    u0 = mms.uexact(z0)
    v0 = mms.uexact_t(z0)
    
    return u0 + v0*t + (t**2)*N_net(z)                      


params = list(N_net.parameters()) +list(a_net.parameters())
opt = torch.optim.LBFGS(params, lr=0.1, max_iter=250,
                        max_eval=None, tolerance_grad=1e-05,
                        tolerance_change=1e-09, history_size=150,
                        line_search_fn='strong_wolfe')

    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5)

toplft, botlft, botrgt, toprgt = [
    torch.tensor([[x, y]]) for (x, y) in [
            (0.0, 25.0),(0.0, 0.0),(25.0, 0.0), (25.0, 25.0)]]


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


def depth_source(z):
    return mms.μ*mms.uexact_y(z) + mms.Z * mms.uexact_t(z)
    
def remote_source(z):
    return mms.μ*mms.uexact_x(z) + mms.Z * mms.uexact_t(z)

def surface_source(z):
    return -mms.μ * mms.uexact_y(z)  

def pde_source(z):
    return mms.uexact_tt(z) - (mms.c**2) * (mms.uexact_xx(z) + mms.uexact_yy(z)) 
           
BC = {}
BC['fault'] = BoundaryCurve(['line', toplft, botlft])
BC['surface'] = BoundaryCurve(['line', botlft, botrgt])
BC['remote'] = BoundaryCurve(['line', botrgt, toprgt])
BC['depth'] = BoundaryCurve(['line', toprgt, toplft])

IBVP = IBVP(SpaceTimeDomain(BC))


IBVP.BC['fault'] = RateFault(IBVP.domain.bdry['fault'], NNet=N,fric_net=a_net, data_function=mms.fault_src)
IBVP.BC['surface'] = BoundaryCondition(IBVP.domain.bdry['surface'], NNet=N, data_function=surface_source, bc='neumann')
IBVP.BC['remote'] = Characteristic(IBVP.domain.bdry['remote'], NNet=N, data_function=remote_source)
IBVP.BC['depth'] = Characteristic(IBVP.domain.bdry['depth'], NNet=N, data_function=depth_source)

'''
IBVP.set_ic(NNet=N, data_function=uexact, bc='primal', label='u0')
IBVP.IC['u0'].t_final = 0.0

IBVP.set_ic(NNet=N, data_function=uexact_t, bc='velocity', label='v0')
IBVP.IC['v0'].t_final = 0.0
'''

IBVP.PDE['wave_eqn'] = PDE(IBVP.domain, NNet=N, data_function=pde_source)


def myloss():#, z_state):#, z01, z02):
    loss_dict = torch.tensor([]).to(device)
    
    loss1 = IBVP.PDE['wave_eqn'].loss().view(1)
    loss_dict = torch.cat([loss_dict, loss1], dim=0)
    
    for k, v in IBVP.BC.items():
        loss1 = v.loss().view(1)
        loss_dict = torch.cat([loss_dict, loss1], dim=0)
     
    #loss_dict = torch.cat([loss_dict, a_net.slope_recovery().view(1)], dim=0) 
    #loss_dict = torch.cat([loss_dict, N_net.slope_recovery().view(1)], dim=0)   
    '''    
    for k in IBVP.IC.values():
        loss1 = k.loss().view(1)
        loss_dict = torch.cat([loss_dict, loss1], dim=0)
    '''  
    
    #weights = LB.get_weights(loss_dict)
    #loss_dict = weights * loss_dict
    
      
        #loss = loss + k.loss()
    loss = loss_dict.sum()
    #loss = loss + a_net.slope_recovery()
    '''        
    l2_lambda = 0.001
    l2_N_reg = torch.tensor(0., requires_grad=True)
    for param in params:
        l2_N_reg = l2_N_reg + torch.norm(param, p=2)
       
    '''
    return loss #+ l2_lambda * l2_N_reg
    
def closure():
    opt.zero_grad(set_to_none=True)
    loss = myloss()

    loss.backward()
    return loss


J = 6
M = 200
K = 24
training_progress = []
training_y = torch.arange(0.0, 25.0, 0.1).unsqueeze(1).to(device)
prev_bestL = 1.0
prev_L = 1.0

net_dict = {}
net_dict['primal'] = N_net
net_dict['state'] = a_net


store_loss = {}
parts = vars(IBVP).copy()
del parts['domain']

for elt in parts:
    for condition in eval("IBVP."+elt):
        store_loss[condition] = []
        
#batch = [6]#, 12, 24, 48]
         
for i in trange(20):
    #J = np.random.choice(batch)      
    #K = 4*J   
    for bdry in IBVP.BC.keys():
        if bdry == 'fault':
            IBVP.BC[bdry].sample_domain(n_samples=J)
            
        else:    
            IBVP.BC[bdry].sample_domain(n_samples=J)
        
    for diffeq in IBVP.PDE.keys():
        IBVP.PDE[diffeq].sample_domain(n_samples=K)
    '''
    for inits in IBVP.IC.keys():
        IBVP.IC[inits].sample_domain(n_samples=M)
    '''
    running_loss = 0.0
    #state_opt.step(state_closure)
    #opt2.zero_grad(set_to_none=True)
    
    #loss = myloss()
    
    #loss.backward()
    #opt2.step()
    opt.step(closure)
    
    L = closure() #- a_net.slope_recovery() - N_net.slope_recovery()
    #state_L = state_closure()
    running_loss += L.item() #+ state_L.item()
    
    scheduler.step(L)
    #state_scheduler.step(state_L)
    #op
    #scheduler.step(L)
    #training_progress.append(a_net(training_y))
    
    #if L < prev_bestL:
    #    torch.save(N.state_dict(), './model_N4.pth')
    #    torch.save(a_net.state_dict(), './model_a4.pth')
    #    prev_bestL = L
     
    training_progress.append(a_net(training_y))
    #print('loss difference:', (L - prev_L).item())
    
    for name in parts:
        for condition in eval("IBVP."+name):
            store_loss[condition].append(eval("IBVP."+name+"[condition].loss()").item())
            
    if i%1 == 0:
        print('running loss:', running_loss)
        print('learning rate:', scheduler.state_dict()['_last_lr'])
        print('-----------------------------')
        
        loss_str = IBVP.loss_report().tolist()
        net_str = IBVP.network_report(net_dict).tolist()
    
        if len(loss_str)<len(net_str):
            diff = len(net_str) - len(loss_str)
            loss_str.extend(["" for i in range(diff)])
        
        else:
            diff = len(loss_str) - len(net_str)
            net_str.extend(["" for i in range(diff)])
         
        report = np.char.add(loss_str, net_str)
    
        for line in report:
            print(line)
    
    prev_L = L


import pickle 

with open('trained_data_inference_hardbc_extradata.pkl', 'rb') as fp:
    data = pickle.load(fp)    
    
sample_size = 1000
z_sample = IBVP.PDE['wave_eqn'].sample_domain(n_samples=sample_size)
err = (N(z_sample) - mms.uexact(z_sample)).square()
err = err.sum()
err = err.sqrt()

e = (mms.uexact(z_sample)).square()
e = e.sum()
e = e.sqrt()

rel_err = err / e

def friction_coeff(y):
    y1 = torch.where(y < 12, -0.005, 0.0)
    y2 = torch.where(y > 17, 0.015, 0.0)
    y3 = torch.where((y >= 12) & (y <= 17), (y - 12) * ((0.015+0.005)/5) - 0.005, 0.0)
    out = y1+y2+y3
        
    return out

zf = IBVP.BC['fault'].sample_domain(n_samples=sample_size)
_, y_sample, _ = zf.split(1, dim=1)

param_err = (a_net(y_sample) - friction_coeff(y_sample)).square()
param_err = param_err.sum()
param_err = param_err.sqrt()

param_e = (friction_coeff(y_sample)).square()
param_e = param_e.sum()
param_e = param_e.sqrt()

rel_param_err = param_err / param_e

for bdry in IBVP.BC.keys():
    if bdry == 'fault':
        IBVP.BC[bdry].sample_domain(n_samples=sample_size)
    else:
        IBVP.BC[bdry].sample_domain(n_samples=sample_size)
        
for diffeq in IBVP.PDE.keys():
    IBVP.PDE[diffeq].sample_domain(n_samples=sample_size)
    '''
    for inits in IBVP.IC.keys():
        IBVP.IC[inits].sample_domain(n_samples=M)
    '''
    
data['wave_eqn'].append(IBVP.PDE['wave_eqn'].loss().cpu().detach().item())
data['fault'].append(IBVP.BC['fault'].loss().cpu().detach().item())
data['surface'].append(IBVP.BC['surface'].loss().cpu().detach().item())
data['depth'].append(IBVP.BC['depth'].loss().cpu().detach().item())
data['remote'].append(IBVP.BC['remote'].loss().cpu().detach().item())

data['L2_abs_err'].append(err.cpu().detach().item())
data['L2_rel_err'].append(rel_err.cpu().detach().item())

data['L2_abs_param'].append(param_err.cpu().detach().item())
data['L2_rel_param'].append(rel_param_err.cpu().detach().item())

with open('trained_data_inference_hardbc_extradata.pkl', 'wb') as fp:
    pickle.dump(data, fp)    
'''
fig, ax = plt.subplots()
for key in store_loss.keys():
    ax.plot(store_loss[key], label=key)
plt.legend()
plt.show()

from matplotlib import colors

y = torch.arange(0, 25, 0.1).unsqueeze(1).to(device)
x = torch.zeros(y.shape).to(device)
t = torch.zeros(y.shape).to(device)
z = torch.cat([x, y, t], dim=1)

fric_net = a_net(y).cpu().detach().numpy()
fric_exact = friction_coeff(z).cpu().detach().numpy()
y = y.cpu().detach().numpy()

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, figsize=(5.5, 7.5))#figsize=(800,500))
#tit = fig.suptitle('Learning Depth-Dependent Friction Parameter\n',
                  # fontsize=14)
#fig.tight_layout(h_pad=10)
#tit.set_y(1.0)
#cmap = plt.get_cmap('plasma_r')
#plt.set_cmap(cmap)
#color_index = np.arange(0, 256, 85).tolist()

ax.set_ylim(25,0)
ax.set_ylabel("Depth (km)")
ax.set_xlabel("(a - b)")
ax.set_xlim(-0.025, 0.025)
ax.plot(fric_exact,y, label="exact",linewidth=2.0, color='black')
#ax.plot(fric_net,y, label="NNet_last",linewidth=2.0)


fig.subplots_adjust(bottom=0.09, left=0.125)
u0 = training_progress[0].cpu().detach().numpy()

cnt = 1

for i in range(len(training_progress)):
    
    
    if i<=10:
        l = training_progress[i].cpu().detach().numpy()
        ax.plot(l[1:], y[1:], label="{:d}".format(i))
        cnt += 1
    
ax.plot(fric_net,y, label="{:d}".format(30), linewidth=2.0)#, color=colors[cnt] )    
ax.legend(bbox_to_anchor=(1.02, 1.1), loc='upper right', ncol=5,
    title='Number of Training Iterations', borderpad=0.25, labelspacing=0.5, fontsize=10)

fig.savefig("./plots/learned_friction_hardbc_datadata(SwRSw).eps" , dpi=300, format='eps')    
'''

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, figsize=(5.5, 5.5))
ax.set_title("Convergence of Loss Components")
ax.set_xlabel('Training Iteration') 
ax.set_ylabel('MSE')
for key in store_loss.keys():
    ax.plot(store_loss[key], label=key, linewidth=2.0)
plt.yscale("log")    
plt.legend()
fig.savefig("./plots/Convergence_of_loss.eps" , dpi=300, format='eps')






     


    
