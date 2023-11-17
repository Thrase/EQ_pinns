import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BoundaryCurve:
    """A class for defining parametric bounding curves.

    ...

    Attributes
    ----------
    ctrl_pts : list
        Coordinates used to parameterize a curve.
    coordinate : tensor
        Stores the most recent set of parameters used to generate data.
    parametric_curve : method
        Stores the parametric equation of the curve.

    Methods
    -------
    _line(t)
        Compute a point on the straight line between two points.
    _spline3(t)
        Compute a point on a Bezier curve with 3 control points.
    __call__(t, reverse=False)
        Compute a point on the parametric curve.
    normal(t)
        Compute the outward unit normal vector to the curve.
    get_point(Nsamples=1)
        Generate data points by sampling the parametric curve.
    """

    def __init__(self, geo_params):
        """Construct class attributes and set parametric equations.

        The parametric curve can be initialized as a straight line
        between two points or as a Bezier curve using three points.

        Parameters
        ----------
            geo_params : list
                A list of  the form geo_params=[str, tensor*] where
                str can either be 'line' or 'spline3' and tensor*
                is either 2 or 3 position tensors, respectively.
                
            n_samples : int
                A number specifying how many data points will be
                sampled from the curve during network training.
        """
        #self.dim = 1
        #self.n_samples = n_samples
        self.ctrl_pts = geo_params[1:]
        self.coordinate = None

        if geo_params[0] == 'line':
            self.parametric_curve = self._line

        elif geo_params[0] == 'Spline3':
            self.parametric_curve = self._spline3

    def _line(self, t):
        """Define a straight-line parametrization between two points."""
        path = self.ctrl_pts[0]*(1-t) + self.ctrl_pts[1]*t

        return path

    def _spline3(self, t):
        """Define a Bezier curve determined by three control points."""
        path = (self.ctrl_pts[1]
                + (self.ctrl_pts[0] - self.ctrl_pts[1]) * (1 - t)**2
                + (self.ctrl_pts[2] - self.ctrl_pts[1]) * (t**2))

        return path

    def __call__(self, t, reverse=False):
        if reverse is True:
            t = torch.tensor([[1.0]]) - t

        return self.parametric_curve(t)

    def normal(self):
        """Compute the outward unit normal to the curve at (x(t), y(t)).

        Returns
        -------
        tensor
            a tensor of normal vectors of shape [dim(t), 2].
        """

        t = self.coordinate.clone().detach().requires_grad_(True)
        x, y = self(t).split(1, dim=1)

        xt, = torch.autograd.grad(x.sum(), t, create_graph=True)
        yt, = torch.autograd.grad(y.sum(), t)
        n = torch.cat([yt, -xt], dim=1).detach()
        n = n / torch.norm(n, dim=1).view(n.shape[0],1)
        
        return n 

    def get_point(self, n_samples, uniform=False):
        """Sample the curve for collocation points.

        Parameters
        ----------
        n_samples : int
            Determines the number of points to sample.
        """
        if uniform is True:
            dt = 1.0 / n_samples
            self.coordinate = torch.arange(0.0, 1.0, dt).unsqueeze(1)
        else:    
            self.coordinate = torch.rand(n_samples, 1)
        z = self(self.coordinate)

        return z




class SpaceTimeDomain:
    def __init__(self, bdry):
        self.bdry = bdry
        #self.n_samples = n_samples
        #self.dim = 2
        #self.t_bounds = [ti, tf]

        self.a, self.b, self.c, self.d = [k for k in self.bdry.keys()]


    def __call__(self, t, s):

        zer = torch.zeros(t.shape)
        one = torch.ones(t.shape)

        bd = (1 - t)*self.bdry[self.a](s, reverse=True)
        bd += t * self.bdry[self.c](s)
        bd += (1-s) * (self.bdry[self.b](t) - ((1-t)*self.bdry[self.b](zer) + t*self.bdry[self.b](one)))
        bd += s * (self.bdry[self.d](t, reverse=True) - ((1-t) * self.bdry[self.d](zer, reverse=True) + t*self.bdry[self.d](one, reverse=True)))
        
        return bd
    
    def spatial_grid(self, Nt=10, Ns=10, time=None):
        Δt = 1/Nt
        Δs = 1/Ns
        
        t = torch.arange(0, 1+Δt, Δt)
        s = torch.arange(0, 1+Δs, Δs)
        
        T, S = torch.meshgrid(t, s, indexing='xy')
    
        grid_vec = torch.stack([k.ravel() for k in (T, S)], axis=1)
        
        t_grid, s_grid = grid_vec.split(1, dim=1)
        
        domain_grid = self(t_grid, s_grid)
        
        if time is not None:
            t0 = torch.tile(time, (len(domain_grid), 1))
        
            domain_grid = torch.cat([domain_grid, t0], dim=1)
        
      
        return domain_grid
        
    def get_mesh(self, row, Nt=10, Ns=10):
        X, Y, Z = torch.swapaxes(row, 0, 1).reshape(3, Ns+1, Nt+1)
        return X, Y, Z
        
    '''
    # Convert from mesh to rows:
    # row_format = np.stack([z.ravel() for z in (xx, yy, data)], axis=1)

    # Convert from rows to mesh:
    # xx, yy, data = np.swapaxes(row_format, 0, 1).reshape(3, len(y), len(x))
    '''
    
    def quadrature_1D(self, f, Δx):
        q = (Δx/3) * (f[0] + 4*f[:, 1:-1:2].sum() + 2*f[: ,2:-1:2].sum() + f[-1])
        return q
        
    def spatial_L2(self, f1, f2, Nt=10, Ns=10, time=0.0):
        
        grid = self.spatial_grid(Nt=Nt, Ns=Ns, time=time).to(device)
        Δx = grid[1,0] - grid[0,0]
        Δy = grid[Nt+1,1] - grid[0,1]
        
        E = torch.tensor([]).to(device)
        
        for j in range(Ns+1):
            sub_grid = grid[j*(Nt+1):(j+1)*(Nt+1)].to(device)
            err = (f1(sub_grid) - f2(sub_grid)).square()    
            err = self.quadrature_1D(err, Δx)
            
            E = torch.cat([E, err], dim=0)
        
        E = E.view(Ns+1, 1)    
        E = self.quadrature_1D(E, Δy)
        E = torch.sqrt(E)
        return E 
        
    def temporal_L2(self, f1, f2, z=None, Nt=10, ti=0.0, tf=1.0):
        T0, T1 = [ti, tf]
        Δt = (T1-T0)/Nt
        t_grid = torch.arange(T0, T1+Δt, Δt)
        t_grid = t_grid.unsqueeze(1)#.to(device)
        
        if z is None:
           t, s = torch.rand(1,2).split(1, dim=1)
           z = self(t, s)#.to(device)
            
        time_extension = [torch.cat([torch.tile(x, (len(t_grid),1)), t_grid], dim=1) for x in z[:]]
        
        ERR = torch.tensor([])#.to(t_grid.device) 
        
        for time_block in time_extension:
            err = (f1(time_block) - f2(time_block)).square()
            err = self.quadrature_1D(err, Δt)
            err = torch.sqrt(err)  
            ERR = torch.cat([ERR, err.view(1,1)], dim=1)
            
        #pt_tensor = torch.tile(z, (len(t_grid), 1)) 
        #pt_tensor = torch.cat([pt_tensor, t_grid], dim=1)
        
        #err = (f1(pt_tensor) - f2(pt_tensor)).square()
        #err = self.quadrature_1D(err, Δt)
        #err = torch.sqrt(err)
        
        ERR = ERR.t()
        z = torch.cat([z, ERR], dim=1)
        return z
        
            
        
        
    
       
    def get_point(self, n_samples, uniform=False):
        
        z = torch.rand((n_samples, 2))
        t, s = z.split(1, dim=1)
        
        '''
        zer = torch.zeros(t.shape)
        one = torch.ones(t.shape)

        bd = (1 - t)*self.bdry['lft'](s)
        bd += t * self.bdry['rgt'](s)
        bd += (1-s) * (self.bdry['bot'](t) - ((1-t)*self.bdry['bot'](zer) + t*self.bdry['bot'](one)))
        bd += s * (self.bdry['top'](t) - ((1-t) * self.bdry['top'](zer) + t*self.bdry['top'](one)))
        '''
        
        pt = self(t, s)
        #τ = t0 * torch.ones((Nsamples, 1)).uniform_(0.0, 1.0)
        
        #pt = torch.cat([pt, τ], dim=1)
        return pt
    
 
    
    
    
    def get_bdry_data(self, name, f=None, t0=1.0, grid=None):
    
        if grid==True:
            Δt = 1 / self.n_samples
            t = torch.arange(0, 1+Δt, Δt)
            t = t.unsqueeze(1)
            
            τ = t0 * torch.arange(self.ti, self.tf+Δt, Δt)
            τ = τ.unsqueeze(1)
            
            pt = self.bdry[name](t)
            
            XT = torch.tensor([])
           # pt_tensor = torch.zeros(pt.shape)
            
            for i in range(len(pt)):
                pt_tensor = torch.tile(pt[i], (len(t), 1)) 
                pt_tensor = torch.cat([pt_tensor, t], dim=1)
                
                XT = torch.cat([XT, pt_tensor], dim=0)
                
            pt = torch.clone(XT)
        else:
            t = torch.rand((self.n_samples, 1))
            
            τ = t0 * torch.ones((self.n_samples, 1)).uniform_(0.0, 1.0)
        
        
            pt = self.bdry[name](t)
            pt = torch.cat([pt, τ], dim=1)
        u = None
        
        if f is not None:
            u = f(pt)
    
        return pt, u
'''
class NetworkViz(SpaceTimeDomain):
    def __init__(self, bdry, Δt, Δs, Δτ, ti=0.0, tf=1.0):
        SpaceTimeDomain.__init__(self, bdry, ti=ti, tf=tf)
        
        self.Δt = Δt
        self.Δs = Δs
        self.Nt = int((1/self.Δt) - 1)
        self.Ns = int((1/self.Δs) - 1) 
        self.τ = torch.arange(ti, tf, Δτ)
        
        self.xy_grid = self.spatial_grid(Nt=self.Nt, Ns=self.Ns, time=None).to(device)
        
    def interpolate_func(self, f, time):
        t = torch.tile(time, (self.xy_grid.shape[0], 1)).to(device)
        
        xyt_grid = torch.cat([self.xy_grid, t], dim=1)
        u = f(xyt_grid)
        xyu_grid = torch.cat([self.xy_grid, u], dim=1)
        return xyu_grid
    
    def change_plot(self, frame_number, plot, f, ax):
        u = self.interpolate_func(f, self.τ[frame_number])
        x, y, z = self.get_mesh(u, Nt=self.Nt, Ns=self.Ns)
        
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, z, cmap="afmhot_r")  

    def an_fig(self, f):
        fps = 30    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        u = self.interpolate_func(f, torch.tensor(0.0))
        x, y, z = self.get_mesh(u, Nt=self.Nt, Ns=self.Ns)

        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        z = z.cpu().detach().numpy()

        plot = [ax.plot_surface(x, y, z, color='0.75', rstride=1, cstride=1)]  

        ani = FuncAnimation(fig, self.change_plot, len(self.τ), fargs= (plot, f, ax), interval=1000 / fps)
        ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))        
'''

