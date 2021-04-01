#!/usr/bin/env python3
""" calculates the maximization step in the EM algorithm for a GMM """

import numpy as np


def maximization(X, g):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors
            for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray):
        return None, None, None
    if not isinstance(g, np.ndarray):
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    if X.ndim != 2 or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if g.shape != (k, n) or X.shape != (n, d):
        return None, None, None
    if n < 1 or d < 1 or k < 1 or k > n:
        return None, None, None
    probs = np.sum(g, axis=0)
    tester = np.ones((n,))
    if not np.isclose(probs, tester).all():
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for ki in range(k):
        density = np.sum(g[ki])
        pi[ki] = density / n
        m[ki] = np.sum(np.matmul(g[ki].reshape(1, n), X), axis=0) / density
        dif = (X - m[ki])
        S[ki] = np.dot(g[ki].reshape(1, n) * dif.T, dif) / density

    return pi, m, S
