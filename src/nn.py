import torch
from torch import Tensor
from . import functions

class NN(object):
    def __init__(self, in_params, out_params) -> None:
        pass

    def foward(self, x):
        pass


class Linear(NN):
    def __init__(self, in_params, out_params, bias=True) -> None:
        super(Linear, self).__init__(in_params, out_params)

        self.inputs = torch.randn(in_params)
        self.weights = torch.randn(out_params, in_params[1])
        if bias:
            self.bias = torch.randn(out_params)

    def foward(self, x):
        return functions.linear(self.inputs, self.weights, self.bias)


