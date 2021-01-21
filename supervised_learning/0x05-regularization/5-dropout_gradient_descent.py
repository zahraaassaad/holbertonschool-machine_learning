#!/usr/bin/env python3

"""updates the weights of a neural network with
   Dropout regularization using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """weights of the network should be
       updated in place"""
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        dw = (1 / len(Y[0])) * np.matmul(dz, A.T)
        db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
        w = "W" + str(i)
        b = "b" + str(i)
        if i != 1:
            d = "D" + str(i - 1)
            da = np.matmul(weights[w].T, dz)
            dz = da * (1 - (A**2)) * (cache[d] / keep_prob)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
