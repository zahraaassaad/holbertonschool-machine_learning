#!/usr/bin/env python3

"""shuffles the data points.
"""

import numpy as np


def shuffle_data(X, Y):
    """Returns: the shuffled X and Y matrices."""
    perm = np.random.permutation
    return perm(X), perm(Y)
