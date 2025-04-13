import torch
import torch.nn as nn

# Borrowed from IRED EBM class
class MLP(nn.Module):
    '''
    Modified from IRED (Simplified?).

    '''
    def __init__(self, inp_dim, out_dim, num_classes, hidden_dim=512):
        super(MLP, self).__init__()
        h = hidden_dim
        
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(inp_dim + out_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim * num_classes)

    def forward(self, x, is_ebm=True):
        '''
        params: 
            is_ebm: bool
            x: Size(batch_size, inp_len+out_len) # already concatenated
            
        return:
            energy (scalar) 
            or latent vector (Size(batch_size, out_len, num_classes))
        '''
            
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        h = swish(self.fc3(h))
        h = self.fc4(h).view(x.size(0), self.out_dim, self.num_classes)

        if is_ebm:
            output = h.pow(2).sum(dim=-1)[..., None] #最后一个维度上平方和，然后添加一个none在最后
        else:
            output = h #raw prediction dist (not normalized)?

        return output
    
    # def parameters(self):#TODO
    #     return super(MLP, self).parameters()