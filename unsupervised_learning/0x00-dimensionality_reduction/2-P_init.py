#!/usr/bin/env python3
"""Function that initializes all variables
required to calculate the P affinities in t-SNE"""

import numpy as np


def P_init(X, perplexity):
    """
    X is a numpy.ndarray of shape (n, d)
      containing the dataset to be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
    perplexity is the perplexity for all Gaussian distributions
    Returns: (D, P, betas, H)
      D: a numpy.ndarray of shape (n, n)
        that calculates the squared pairwise distance between two data points
      The diagonal of D should be 0s
      P: a numpy.ndarray of shape (n, n)
        initialized to all 0‘s that will contain the P affinities
      betas: a numpy.ndarray of shape (n, 1)
        initialized to all 1’s that will contain all of the beta values
      \beta_{i} = \frac{1}{2{\sigma_{i}}^{2} }
      H is the Shannon entropy for perplexity perplexity with a base of 2
    """
    n = X.shape[0]
    X_sum = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), X_sum).T, X_sum)
    np.fill_diagonal(D, 0.)
    betas = np.ones((n, 1))
    P = np.zeros((n, n))
    P_sum = sum(P)
    H = np.log2(perplexity)
    return (D, P, betas, H)
