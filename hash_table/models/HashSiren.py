
import torch
from torch import nn
import numpy as np

class HashSiren(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30.0):

        super().__init__()
        self.hash_mod = hash_mod

        self.table = torch.nn.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, world_size):
        # self.table.data = self.table.data.reshape(world_size[0], world_size[1], world_size[2], -1)
        if self.hash_mod:
            # output = self.net(self.table[:,:])
            # output = self.net(self.table[coords[:,0].tolist(),coords[:,1].tolist(),coords[:,2].tolist(),:])
            output = self.net(self.table[coords])
        else:
            output = self.net(coords)
        return output


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out
    
    
    
class HashMlp(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True):

        super().__init__()
        self.hash_mod = hash_mod

        self.table = torch.nn.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(MlpLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(MlpLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            self.net.append(final_linear)
        else:
            self.net.append(MlpLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, world_size):
        # self.table.data = self.table.data.reshape(world_size[0], world_size[1], world_size[2], -1)
        if self.hash_mod:
            # output = self.net(self.table[:,:])
            # output = self.net(self.table[coords[:,0].tolist(),coords[:,1].tolist(),coords[:,2].tolist(),:])
            output = self.net(self.table[coords])
        else:
            output = self.net(coords)
        # output = torch.clamp(output, min = -1.0,max = 1.0)

        return output
class HashMlp1(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True):

        super().__init__()
        self.hash_mod = hash_mod

        self.table = torch.nn.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(MlpLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(MlpLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            self.net.append(final_linear)
        else:
            self.net.append(MlpLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # self.table.data = self.table.data.reshape(world_size[0], world_size[1], world_size[2], -1)
        if self.hash_mod:
            output = self.net(self.table[:,:])
            # output = self.net(self.table[coords[:,0].tolist(),coords[:,1].tolist(),coords[:,2].tolist(),:])
        else:
            output = self.net(coords)
        # output = torch.clamp(output, min = -1.0,max = 1.0)

        return output



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(MlpLayer(hidden_features, out_features))

        for i in range(hidden_layers):
            self.net.append(MlpLayer(hidden_features, out_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(MlpLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class MlpLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        activate_f = nn.ReLU()
        out = activate_f(self.linear(input))
        return out


class VoxelGrid(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True):

        super().__init__()
        self.hash_mod = hash_mod

        self.table = torch.nn.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
    
    def forward(self, coords):
        if self.hash_mod:
            output = self.table
        # output = torch.clamp(output, min = -1.0,max = 1.0)

        return output

class grid_feature(nn.Module):
    def __init__(self,
                 hash_mod,
                 hash_table_length, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 outermost_linear=True):

        super().__init__()
        self.hash_mod = hash_mod

        self.table = torch.nn.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(MlpLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(MlpLayer(hidden_features, hidden_features))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            self.net.append(final_linear)
        else:
            self.net.append(MlpLayer(hidden_features, out_features))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # self.table.data = self.table.data.reshape(world_size[0], world_size[1], world_size[2], -1)
        if self.hash_mod:
            output = self.net(self.table[:,:])
            # output = self.net(self.table[coords[:,0].tolist(),coords[:,1].tolist(),coords[:,2].tolist(),:])
        else:
            output = self.net(coords)
        # output = torch.clamp(output, min = -1.0,max = 1.0)

        return output


if __name__ == '__main__':
    print('start')
    hashsiren = HashSiren(
        hash_mod = True,
        hash_table_length = 10, 
        in_features = 2, 
        hidden_features = 2, 
        hidden_layers = 2, 
        out_features  = 1,
    )