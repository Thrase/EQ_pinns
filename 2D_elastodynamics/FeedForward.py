import torch
import torch.nn as nn
import numpy as np

class Feedforward(torch.nn.Module):
    """Compute the forward pass of a Feed-Forward Neural Net."""

    def __init__(self, layers, activations):
        """Initialize the network."""
        super().__init__()
        self.layers = layers
        self.activation = nn.ModuleList([act for act in activations])
        self.softplus = nn.Softplus()
        self.mse = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
                                      
        self.hidden_dim = len(self.layers) - 2
        self.a = torch.nn.Parameter(torch.rand(self.hidden_dim, 1))
            
        for i in range(len(self.layers)-1):
            nn.init.xavier_uniform_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, z):
        """Compute the network's forward pass."""
        for i in range(self.hidden_dim):
            w = self.linears[i](z)
            z = self.activation[i](w)
        z = self.linears[-1](z)
        return z
    
    '''    
    def slope_recovery(self):
        rec = torch.exp(self.a)   
        rec = rec.sum()
        rec = (1/self.hidden_dim) * rec
        rec = 1 / rec
        
        return rec
    ''' 
    def slope_recovery(self):
    
        R = torch.tensor([]).to("cuda")
        
        for p in self.a:
            N = len(p)
            rec = p.sum() / N
            rec = torch.exp(rec)
            rec = rec.view(1)
            R = torch.cat([R, rec], dim=0)
            
        rec = R.sum()
        rec = (1/self.hidden_dim) * rec
        rec = 1 / rec
        
        return rec
             
        
class FeedforwardAdapt(torch.nn.Module):
    """Compute the forward pass of a Feed-Forward Neural Net."""

    def __init__(self, layers, activations, adaptive_weight=None):
        """Initialize the network."""
        super().__init__()
        self.layers = layers
        self.activation = nn.ModuleList([act for act in activations])
        self.softplus = nn.Softplus()
        self.mse = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1])
                                      for i in range(len(layers)-1)])
                                      
        self.hidden_dim = len(self.layers) - 2
        self.b = torch.nn.Parameter(torch.rand(1, 1))
        self.n = 10.0  # adaptive weight scaling
        
        if adaptive_weight=='layer':
            self.a = torch.nn.Parameter(torch.ones(self.hidden_dim, 1) / self.n)
        elif adaptive_weight=='neuron':
            self.a = torch.nn.Parameter(torch.ones(self.hidden_dim, self.layers[1]) / self.n)
        else:
            self.a = np.ones(self.hidden_dim,1)
            
        for i in range(len(self.layers)-1):
            nn.init.xavier_uniform_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, z):
        """Compute the network's forward pass."""
        for i in range(self.hidden_dim):
            w = self.linears[i](z)
            w = self.n * self.a[i] * w
            z = self.activation[i](w)
        z = self.linears[-1](z)
        return z
    
    '''    
    def slope_recovery(self):
        rec = torch.exp(self.a)   
        rec = rec.sum()
        rec = (1/self.hidden_dim) * rec
        rec = 1 / rec
        
        return rec
    ''' 
    def slope_recovery(self):
    
        R = torch.tensor([]).to("cuda")
        
        for p in self.a:
            N = len(p)
            rec = p.sum() / N
            rec = torch.exp(rec)
            rec = rec.view(1)
            R = torch.cat([R, rec], dim=0)
            
        rec = R.sum()
        rec = (1/self.hidden_dim) * rec
        rec = 1 / rec
        
        return rec
