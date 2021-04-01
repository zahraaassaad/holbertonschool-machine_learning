#!/usr/bin/env python3
"""Function that tests for the optimum number of clusters by variance"""

import numpy as np
variance = __import__('2-variance').variance
kmeans = __import__('1-kmeans').kmeans


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum
    number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum
    number of clusters to check for (inclusive)
    iterations is a positive integer containing
    the maximum number of iterations for K-means
    This function should analyze at least 2 different cluster sizes
    You should use:
        kmeans = __import__('1-kmeans').kmeans
        variance = __import__('2-variance').variance
    You may use at most 2 loops
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs
        of K-means for each cluster size
        d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax and not isinstance(kmax, int) or kmax < 1:
        return None, None
    if kmax and kmin >= kmax:
        return None, None

    n, d = X.shape
    kmax = n if not kmax else kmax
    results = []
    d_vars = []
    for clusters in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, clusters, iterations=1000)
        results.append((centroids, clss))
        if clusters == kmin:
            comp_var = variance(X, centroids)
        var = variance(X, centroids)
        d_vars.append(comp_var - var)

    return results, d_vars
