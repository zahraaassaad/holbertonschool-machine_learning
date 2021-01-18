#!/usr/bin/env python3

"""calculates the sensitivity for each class"""

import numpy as np


def sensitivity(confusion):
    """returns the sensitivity of each class."""
    return np.sum((confusion * np.identity(
        confusion.shape[0])) / np.sum(confusion, axis=1), axis=1)
