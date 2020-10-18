#!/bin/python

import torch
from torch import Tensor

# Matrix multiplication
## The number of columns of A-matrix must be equals B-matrix number of rows
## https://en.wikipedia.org/wiki/Matrix_multiplication
def matmul(a, b: Tensor) -> Tensor:
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i, j] += (a[i, :] * b[:, j]).sum()
    return c

# Standard score is the number of standard deviations by which the value 
# of a raw score is above or below the mean of what is being observed.
# https://en.wikipedia.org/wiki/Standard_score
# z = (x - mean) / std
def normalize(x, m, s: Tensor) -> Tensor:
    return (x-m)//s


def linear(x, w, b: Tensor) -> Tensor:
    output = matmul(x, w.t())
    if b is not None:
        output += b
    return output
