#!/usr/bin/env python3

"""Calculates the normalization constants of a matrix."""
import numpy as np


def normalization_constants(X):
    """Returns: the mean and standard deviation
    """
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    return mean, stdev 
