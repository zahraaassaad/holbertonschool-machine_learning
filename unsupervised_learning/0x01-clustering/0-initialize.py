#!/usr/bin/env python3
"""initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """
    Returns: a numpy.ndarray of shape (k, d)
        containing the initialized centroids for each cluster,
        or None on failure
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0:
        return None
    n, d = np.shape(X)
    l = np.min(X, axis=0)
    h = np.max(X, axis=0)
    centroids = np.random.uniform(l, h, size=(k, d))
    return centroids
