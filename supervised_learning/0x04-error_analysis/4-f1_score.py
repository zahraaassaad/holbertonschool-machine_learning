#!/usr/bin/env python3

"""calculates the F1 score"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """returns F1 score of each class."""
    return 2 * (sensitivity(confusion) * precision(confusion)) / (
        sensitivity(confusion) + precision(confusion))
