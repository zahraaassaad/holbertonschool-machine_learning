#!/usr/bin/env python3
import numpy

"""
0-mean_cov.py
calculates the mean and covariance of a data set
"""


def mean_cov(X):
    """Returns: mean, cov"""
    if type(X) != numpy.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X) < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.array([sum(X)/len(X)])
    rows, cols = X.shape
    cov_mat = np.zeros((cols, cols))
    for i in range(cols):
        for j in range(cols):
            # store the value in the matrix
            mean_x, mean_y = X[:, i].mean(), X[:, j].mean()
            n = len(X[:, i])
            cov_mat[i][j] = sum((X[:, i] - mean_x) * (X[:, j] - mean_y)) / n
    return mean, cov_mat
