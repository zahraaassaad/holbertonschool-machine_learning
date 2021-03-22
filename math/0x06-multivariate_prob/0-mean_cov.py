#!/usr/bin/env python3
"""
0-mean_cov.py
calculates the mean and covariance of a data set
"""
import numpy


def mean_cov(X):
    """Returns: mean, cov"""
    if type(X) != numpy.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X) < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0)
    rows, cols = X.shape
    a = X - mean
    c = np.matmul(a.T, a)
    cov = c/(rows - 1)
    return mean.reshape((1, d)), cov
