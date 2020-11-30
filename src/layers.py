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
        self.x = x
        return (self.x @ self.weights.t()) + self.bias

    def backward(self):
        self.x.grad = self.weights.t() @ self.loss_grad
        self.weights.grad = self.x @ self.loss_grad
        self.bias.grad = self.loss_grad.sum(0)

    def __call__(self, x):
        return self.foward(x)

class ReLU():
    def __init__(self, x):
        self.x = x
        self.foward()

    def foward(self, x):
        self.out = self.x.clamp_min(0.)
        return self.out

    def backward(self):
        self.x.grad = (self.x > 0).float() * self.loss_grad


class MSE():
    def __init__(self, x, target):
        self.x = x


    def __call__(self, x):
        return self.foward(x)
