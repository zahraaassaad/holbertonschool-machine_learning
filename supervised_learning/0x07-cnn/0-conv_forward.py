#!/usr/bin/env python3
""" Module to compute forward propagation in a convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Convolution Forward propagation"""
    m, h, w, c = A_prev.shape
    kh, kw, kc, nc = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:
        ph, pw = 0, 0
    pad_result = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    conv_h = (h - kh + 2 * ph) // sh + 1
    conv_w = (w - kw + 2 * pw) // sw + 1
    conv_result = np.zeros((m, conv_h, conv_w, nc))
    for row in range(conv_h):
        for col in range(conv_w):
            for channel in range(nc):
                conv_result[:, row, col, channel] = np.sum(
                    pad_result[:, row * sh:(kh + (row * sh)),
                               col * sw:(kw + (col * sw))]
                    * W[:, :, :, channel],
                    axis=(1, 2, 3)
                )
                conv_result[:, row, col, channel] = activation(
                    conv_result[:, row, col, channel] + b[:, :, :, channel])
    return conv_result
