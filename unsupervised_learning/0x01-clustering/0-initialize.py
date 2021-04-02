#!/usr/bin/env python3
"""Initialize function"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n: the number of data points
           d: the number of dimensions for each data point
        k: positive integer containing the number of clusters
    Returns: numpy.ndarray of shape (k, d) containing the initialized
             centroids for each cluster, or None on failure
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0:
        return None
    n, d = np.shape(X)
    l = np.min(X, axis=0)
    h = np.max(X, axis=0)
    return np.random.uniform(l, h, size=(k, d))
