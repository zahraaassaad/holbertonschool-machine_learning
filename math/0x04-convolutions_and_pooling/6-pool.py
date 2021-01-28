#!/usr/bin/env python3
"""performs pooling of images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """returns pooled images"""
    hk = kernel_shape[0]
    wk = kernel_shape[1]
    nc = images.shape[3]
    m = images.shape[0]
    hm = images.shape[1]
    wm = images.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    out_h = int((hm - hk) / st0) + 1
    out_w = int((wm - wk) / st1) + 1
    pooled = np.zeros((m, out_h, out_w, nc))
    img = images.copy()
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h * st0: h * st0 + hk, w * st1: w * st1 + wk, :]
            if mode == "max":
                v = np.max(matrix, axis=(1, 2))
            else:
                v = np.average(matrix, axis=(1, 2))
            pooled[:, h, w, :] = v
    return(pooled)
