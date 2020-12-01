import torch
import numpy as np

class Layer:
    def __call__(self, *args):
        self.args = args
        self.out = self.foward(*args)
        return self.out

    def foward(self):
        raise Exception("Not implemented")
    def __backward(self):
        raise Exception("Not implemented")
    def backward(self):
        return self.__backward(self.out, *args)

class Linear(Layer):
    def foward(self, x):
        return x @ self.w.t() + self.bias

    def __backward(self, out, x):
        x.g = self.w.t() @ out.g 
        self.w.g = x.t() @ out.g
        self.bias.g = out.g.sum(0)
    
class ReLU():
    def foward(self, x):
        return self.x.clamp_min(0.)

    def __backward(self, out, x):
        x.g = (x > 0).float() * out.g

class MSE():
    def foward(self, x, targets):
        return (x.squeeze() - targets).pow(2).mean()

    def __backward(self, out, x, targets):
        x.g = 2. * (x.squeeze() - targets).unsqueeze(-1) / targets.shape[0]

