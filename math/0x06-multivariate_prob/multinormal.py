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

    def pdf(self, x):
        """
        public instance method def pdf(self, x):
            calculates the PDF at a data point:
        x is a numpy.ndarray of shape (d, 1):
            containing the data point whose PDF should be calculated
        d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray:
          raise a TypeError with the message x must be a numpy.ndarray
        If x is not of shape (d, 1):
          raise a ValueError with the message x must have the shape ({d}, 1)
        Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        cov = self.cov
        if len(x.shape) is not 2 or x.shape[1] is not 1\
                or x.shape[0] is not cov.shape[0]:
            raise ValueError("x must have the shape ({}, 1)".
                             format(cov.shape[0]))
        cov_inv = np.linalg.inv(cov)
        mean = self.mean
        cov_det = np.linalg.det(cov)
        density = np.sqrt(np.power((2 * np.pi), cov.shape[0]) * cov_det)
        y = np.matmul((x - mean).T, cov_inv)
        pdf = (1 / density) * np.exp(-1 * np.matmul(y, (x - mean)) / 2)
        return pdf.reshape(-1)[0]
