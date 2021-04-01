#!/usr/bin/env python3
""" Calculates the probability density function of a Gaussian distribution """
import numpy as np


def pdf(X, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
        should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    You are not allowed to use any loops
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
            data point
    All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    d = S.shape[0]
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    part_1 = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    part_2 = np.dot((X - m), inv)
    part_3 = np.sum(-part_2 * (X - m) / 2, axis=1)
    pdf = part_1 * np.exp(part_3)

    return np.where(pdf < 1e-300, 1e-300, pdf)
