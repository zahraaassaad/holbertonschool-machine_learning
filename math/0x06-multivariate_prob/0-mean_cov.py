#!/usr/bin/env python3
"""
0-mean_cov.py
calculates the mean and covariance of a data set
"""
import numpy


def mean_cov(X):
    """Returns: mean, cov"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    d = X.shape[1]
    mean = np.mean(X, axis=0)
    a = X - mean
    c = np.matmul(a.T, a)
    cov = c/(X.shape[0] - 1)
    return mean.reshape((1, d)), cov
