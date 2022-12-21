import torch
from torch import nn
from torch.autograd import Variable
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               
class MSELoss1(nn.Module):
    def __init__(self):
        super(MSELoss1, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)


        return loss

loss_dict = {'mse': MSELoss,
             'mse1': MSELoss1}


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class TVLoss_3(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss_3,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,grid):
        batch_size = grid.size()[0]
        D = grid.size()[2]
        H = grid.size()[3]
        W = grid.size()[4]
        
        count_d = self._tensor_size(grid[:,:,1:,:,:])
        count_h = self._tensor_size(grid[:,:,:,1:,:])
        count_w = self._tensor_size(grid[:,:,:,:,1:])
        
        d_tv = torch.pow((grid[:,:,1:,:,:]-grid[:,:,:D-1,:,:]),2).sum()
        h_tv = torch.pow((grid[:,:,:,1:,:]-grid[:,:,:,:H-1,:]),2).sum()
        w_tv = torch.pow((grid[:,:,:,:,1:]-grid[:,:,:,:,:W-1]),2).sum()
        return self.TVLoss_weight*3*(d_tv/count_d+h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]
    
if __name__ == '__main__':
    TVloss = TVLoss()
    a = torch.rand((2,3,4,5))
    b = TVloss(a)