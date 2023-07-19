import torch
import numpy as np

class StrikeSlipMMS():
    def __init__(self, w, μ=32, ρ=2.670, a=0.015, b=0.02, L=2.0, f0=0.6, 
            σn=50.0, v0 =  10e-6):
            
        self.w = w
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
        
    def first_partials(self, z):
        grad, = torch.autograd.grad(self.w(z).sum(), z, create_graph=True)
        u, v = grad.split(1,dim=1)  # gradient can be split into parts
        
        return u, v
        
    def state_variable(self, z):
        u, v = self.first_partials(z)
        n = -1
        
        return self.a * torch.log( (self.v0/v) * torch.sinh((n*self.μ * u) / (self.a * self.σn)))
      
        
    def state_evolution(self, z):
        ψ = self.state_variable(z)
        _, v = self.first_partials(z)
        
        
        return (self.b * self.v0 / self.L) * torch.exp(((self.f0-ψ)/self.b) - (2 * v.abs() / self.v0))
        
    def ageing_source(self, z):
        grad, = torch.autograd.grad(self.state_variable(z).sum(), z, create_graph=True)
        _, dt = grad.split(1,dim=1)
        
        return (dt - self.state_evolution(z)).detach()
        
    def fault_friction(self, z):
        x, t = z.split(1, dim=1) 
        
        _, v = self.first_partials(z)
        
        return self.a * torch.arcsinh((v/self.v0) * torch.exp(self.state_variable(z) / self.a))
        
    def fault_source(self, z):    
        u, _ = self.first_partials(z)
        
        return ((self.μ * u) - (self.σn * self.fault_friction(z))).detach()
        
        
        
z = torch.rand((1,2), requires_grad=True)
