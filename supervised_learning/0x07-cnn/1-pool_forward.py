#!/usr/bin/env python3
""" Compute forward propagation in a convolutional layer with pooling"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Forward prop convolution"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    conv_h = (h - kh) // sh + 1
    conv_w = (w - kw) // sw + 1
    conv_result = np.zeros((m, conv_h, conv_w, c))
    for row in range(conv_h):
        for col in range(conv_w):
            for ch in range(c):
                if mode == 'max':
                    conv_result[:, row, col, ch] = np.max(
                        A_prev[:, row * sh:kh + row * sh,
                               col * sw:kw + col * sw, ch],
                        axis=(1, 2)
                    )
                else:
                    conv_result[:, row, col, ch] = np.mean(
                        A_prev[:, row * sh:kh + row * sh,
                               col * sw:kw + col * sw, ch],
                        axis=(1, 2)
                    )
    return conv_result
