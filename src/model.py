import layers
import functions
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(
                layers.Linear(n_in, nh), 
                layers.ReLU(),
                layers.Linear(nh, n_out))
        self.loss = functions.MSE



if __name__ == '__main__':
    model = Model(10, 5, 2)
    print(model.layers)
