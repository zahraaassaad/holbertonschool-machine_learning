#!/usr/bin/env python3

"""
calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Returns: the cost of the network"""
    l2 = 0
    for i in range(1, L+1):
        w = "W" + str(i)
        l2 += 1/m * lambtha/2 * np.linalg.norm(weights[w])
    return cost + l2
