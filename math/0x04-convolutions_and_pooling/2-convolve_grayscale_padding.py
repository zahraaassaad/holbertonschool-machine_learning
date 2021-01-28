#!/usr/bin/env python3

"""
performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    (ph, pw) = padding
    padded_images = np.pad(
        images,
        [(0, 0), (ph, ph), (pw, pw)]
    )
    m, h, w = padded_images.shape
    conv_shape = (m, h - kh + 1, w - kw + 1)
    conv = np.zeros(conv_shape)
    for row in range(conv_shape[1]):
        for column in range(conv_shape[2]):
            sub_matrix = padded_images[:, row: row + kh, column: column + kw]
            conv[:, row, column] = (sub_matrix * kernel).sum(axis=(1, 2))
    return conv
