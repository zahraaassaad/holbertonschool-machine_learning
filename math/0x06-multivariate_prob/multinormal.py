#!/usr/bin/env python3
"""MultiNormal class"""
import numpy as np


class MultiNormal:
    """ Multivariate Normal distribution"""

    def __init__(self, data):
        """
        class constructor def __init__(self, data):
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray:
          raise a TypeError: data must be a 2D numpy.ndarray
        If n is less than 2:
          raise a ValueError: data must contain multiple data points
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) is not 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        X = data.T
        d = data.shape[0]
        mean = np.mean(X, axis=0)
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        a = X - mean
        C = np.matmul(a.T, a)
        self.cov = C / (X.shape[0] - 1)
