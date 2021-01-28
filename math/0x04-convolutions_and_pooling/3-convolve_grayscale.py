#!/usr/bin/env python3

"""
performs a convolution on images with channels
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    sh, sw = stride
    if padding in ('same', 'valid'):
        if padding == 'same':
            (m, h, w) = images.shape
            pw = int(np.ceil((((w - 1) * sw) + kw - w) / 2))
            ph = int(np.ceil((((h - 1) * sh) + kh - h) / 2))
        else:
            (ph, pw) = (0, 0)
    else:
        (ph, pw) = padding

    padded_images = np.pad(
        images,
        [(0, 0), (ph, ph), (pw, pw)]
    )

    (m, h, w) = padded_images.shape
    output_shape = (m, int(((h - kh) / sh) + 1), int(((w - kw) / sw) + 1))

    output = np.zeros(output_shape)

    for row in range(output_shape[1]):
        for column in range(output_shape[2]):
            sub_matrix = padded_images[
                :,
                (row * sh): (row * sh) + kh,
                (column * sw): (column * sw) + kw
            ]
            output[:, row, column] = (sub_matrix * kernel).sum(axis=(1, 2))

    return output
