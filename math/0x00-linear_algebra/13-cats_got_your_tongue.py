#!/usr/bin/env python3

"""
Module for np_cat.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate 2 np arrays."""
    return(np.concatenate((mat1, mat2), axis=axis))
