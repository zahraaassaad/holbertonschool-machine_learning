#!/usr/bin/env python3

"""normalizes (standardizes) a matrix.
"""

import numpy as np


def normalize(X, m, s):
    """The normalized X matrix."""
    X = (X - m) / s
    return X
