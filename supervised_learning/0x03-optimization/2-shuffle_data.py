#!/usr/bin/env python3

"""shuffles the data points.
"""

import numpy as np


def shuffle_data(X, Y):
    """Returns: the shuffled X and Y matrices."""
    X_new = np.random.permutation(X)
    Y_new = np.random.permutation(Y)
    return X_new, Y_new
