import torch
import torch.nn as nn
from math import sqrt



class Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))
        
    def forward(self, x):
        return LinearFunction.apply(x, self.weights, self.bias)


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return ReluFunction.apply(x)
