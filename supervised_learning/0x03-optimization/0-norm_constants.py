#!/usr/bin/env python3

"""Calculates the normalization constants of a matrix."""

import numpy as np


def normalization_constants(X):
    mean = np.mean(X)
    stdev = np.std(X)
    return mean, stdev
