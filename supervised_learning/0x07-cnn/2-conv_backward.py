#!/usr/bin/env python3
""" Module to compute back propagation in a convolutional layer with padding"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Back prop for NN layer"""
    m, h_prev, w_prev, _ = A_prev.shape
    _, h_new, w_new, _ = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((sh * h_prev) - sh + kh - h_prev) // 2
        pw = ((sw * w_prev) - sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    pad_img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(pad_img.shape)
    dW = np.zeros(W.shape)

    for img in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    a = row * sh
                    b = a + kh
                    c = col * sw
                    d = c + kw
                    dA[img, a:b, c:d, :] += dZ[img, row, col, ch]\
                        * W[:, :, :, ch]
                    dW[:, :, :, ch] += pad_img[img, a:b, c:d, :]\
                        * dZ[img, row, col, ch]
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dW, db
