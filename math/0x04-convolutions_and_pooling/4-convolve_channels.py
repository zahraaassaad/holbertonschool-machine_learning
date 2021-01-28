#!/usr/bin/env python3
"""performs convolution of images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """returns convolved images"""
    hk = kernel.shape[0]
    wk = kernel.shape[1]
    m = images.shape[0]
    hm = images.shape[1]
    wm = images.shape[2]
    st1 = stride[1]
    st0 = stride[0]
    if padding == "valid":
        pad0 = 0
        pad1 = 0
    elif padding == "same":
        pad0 = int(((hm - 1) * st0 + hk - hm) / 2) + 1
        pad1 = int(((wm - 1) * st1 + wk - wm) / 2) + 1
    else:
        pad0 = padding[0]
        pad1 = padding[1]
    out_h = int((hm + 2 * pad0 - hk) / st0) + 1
    out_w = int((wm + 2 * pad1 - wk) / st1) + 1
    convoluted = np.zeros((m, out_h, out_w))
    img = np.pad(images, ((0, 0), (pad0, pad0), (
                           pad1, pad1), (0, 0)), 'constant')
    for h in range(out_h):
        for w in range(out_w):
            matrix = img[:, h * st0: h * st0 + hk, w * st1: w * st1 + wk, :]
            v = np.sum(matrix * kernel, axis=1).sum(axis=1).sum(axis=1)
            convoluted[:, h, w] = v
    return(convoluted)
