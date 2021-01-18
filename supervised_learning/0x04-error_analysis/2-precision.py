#!/usr/bin/env python3

"""calculates the precision for each class"""

import numpy as np


def precision(confusion):
    """returns precision of each class."""
    return np.sum((confusion * np.identity(
        confusion.shape[0])) / np.sum(confusion, axis=0), axis=1)
