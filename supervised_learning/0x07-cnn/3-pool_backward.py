#!/usr/bin/env python3
"""performs back propagation over a pooling
   layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """returns partial derivatives with respect to the previous
       layer (dA_prev)"""
    hk = kernel_shape[0]
    wk = kernel_shape[1]
    nc = A_prev.shape[3]
    m = A_prev.shape[0]
    hm = A_prev.shape[1]
    wm = A_prev.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    out_h = int((hm - hk) / st0) + 1
    out_w = int((wm - wk) / st1) + 1
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(nc):
                    matrix = A_prev[i, h * st0: h * st0 + hk,
                                    w * st1: w * st1 + wk, c]
                    if mode == "max":
                        max_va = matrix.max() == matrix
                        dA_prev[i, h * st0: h * st0 + hk,
                                w * st1: w * st1 + wk,
                                c] += dA[i, h, w, c] * max_va
                    else:
                        dA_prev[i, h * st0: h * st0 + hk,
                                w * st1: w * st1 + wk,
                                c] += dA[i, h, w, c] / (hk * wk)
    return(dA_prev)
