#!/usr/bin/env python3
"""conducts forward propagation
   using Dropout:"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Returns: outputs of each layer and
       the dropout mask used on each layer"""
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        w = "W" + str(i)
        b = "b" + str(i)
        a = "A" + str(i - 1)
        d = "D" + str(i)
        Z = np.matmul(weights[w],
                      cache[a]) + weights[b]
        a_new = "A" + str(i)
        if i != L:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            true_false_matrix = np.random.rand(
                A.shape[0], A.shape[1]) < keep_prob
            cache[d] = np.multiply(true_false_matrix, 1)
            a_temp = np.multiply(A, cache[d])
            cache[a_new] = a_temp / keep_prob

        else:
            t = np.exp(Z)
            a_new = "A" + str(i)
            cache[a_new] = t / t.sum(axis=0, keepdims=True)
    return (cache)
