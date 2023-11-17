import torch
import numpy as np
import torch.autograd.forward_ad as fwAD

        
class StrikeSlipMMS():
    def __init__(self, μ=32, ρ=2.670, a=0.015, b=0.02, L=2.0, f0=0.6, 
            σn=50.0, v0=10e-6):
            
        self.μ = μ
        self.ρ = ρ
        self.c = np.sqrt(self.μ / self.ρ) 
        self.Z = np.sqrt(self.μ * self.ρ)
        self.a = a
        self.b = b
        self.L = L
        self.f0 = f0
        self.σn = σn
        self.v0 = v0
    
    def uexact(self, z):
        x, y, t = z.split(1, dim=1)
        
        return torch.tanh((x + y + self.c*t)/20)
    
    def uexact_t(self, z):
    
        if z.requires_grad is True:
            z1 = z
            trace = True
        else:
            z1 = z.clone().detach().requires_grad_(True)
            trace = False
            
        out = self.uexact(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1, create_graph=trace)
        _, _, dt = grad.split(1, dim=1)
         
        return dt
    
    def uexact_x(self, z):
        if z.requires_grad is True:
            z1 = z
            trace = True
        else:
            z1 = z.clone().detach().requires_grad_(True)
            trace = False
            
        out = self.uexact(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1, create_graph=trace)
        dx, _, _ = grad.split(1, dim=1)
         
        return dx
        
    def uexact_y(self, z):
        if z.requires_grad is True:
            z1 = z
            trace = True
        else:
            z1 = z.clone().detach().requires_grad_(True)
            trace = False
            
        out = self.uexact(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1, create_graph=trace)
        _, dy, _ = grad.split(1, dim=1)
         
        return dy
    
    def uexact_xt(self, z):
        z1 = z.clone().detach().requires_grad_(True)
        out = self.uexact_t(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1)
        dx, _, _ = grad.split(1, dim=1)
         
        return dx
    
    def uexact_tt(self, z):
        z1 = z.clone().detach().requires_grad_(True)
        out = self.uexact_t(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1)
        _, _, dt = grad.split(1, dim=1)
         
        return dt
        
    def uexact_xx(self, z):
        z1 = z.clone().detach().requires_grad_(True)
        out = self.uexact_x(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1)
        dx, _, _ = grad.split(1, dim=1)
         
        return dx
        
    def uexact_yy(self, z):
        z1 = z.clone().detach().requires_grad_(True)
        out = self.uexact_y(z1)
        out = out.sum()
         
        grad, = torch.autograd.grad(out, z1)
        _, dy, _ = grad.split(1, dim=1)
         
        return dy
    
     
    
    def friction_coeff(self, z):
        x, y, t = z.split(1, dim=1)
    
    
        y1 = torch.where(y < 12, -0.005, 0.0)
        y2 = torch.where(y > 17, 0.015, 0.0)
        y3 = torch.where((y >= 12) & (y <= 17), (y - 12) * (0.02/5) - 0.005, 0.0)
        out = y1+y2+y3
        
        return out
    
    def fe_rate(self, z):
        return self.f0 + self.friction_coeff(z)*torch.log(2*self.uexact_t(z) / (self.v0))

    def fe(self, z):
        x, y, t = z.split(1, dim=1) 
        return self.a * torch.arcsinh((self.uexact_t(z)/self.v0) * torch.exp(self.state_exact(z) / self.a))
        
    def fault_src(self, z):
        x, y, t = z.split(1, dim=1)
        
        return -self.μ*self.uexact_x(z) - self.σn*self.fe_rate(z)

