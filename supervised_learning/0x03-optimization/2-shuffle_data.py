#!/usr/bin/env python3

"""shuffles the data points.
"""

import numpy as np


def shuffle_data(X, Y):
    """Returns: the shuffled X and Y matrices."""
    perm = X.shape[0]
    shuff_op = np.random.permutation(perm)
    return X[shuff_op], Y[shuff_op]
