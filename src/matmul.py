#!/bin/python

import torch

def matmul(a, b: torch.tensor) -> torch.tensor:
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i, j] += (a[i, :] * b[:, j]).sum()
    return c

if __name__ == '__main__':
    m1 = torch.randn((10, 28*28))
    m2 = torch.randn((28*28, 20))
    print(matmul(m1, m2))
