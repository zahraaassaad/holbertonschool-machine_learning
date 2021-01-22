#!/usr/bin/env python3

"""
calculates the normalization constants
"""

import numpy as np


def normalization_constants(X):
    """Returns: the mean and standard deviation"""
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
