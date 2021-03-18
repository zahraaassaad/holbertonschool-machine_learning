#!/usr/bin/env python3

"""
5-definiteness.py
calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Return: the string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite if
        the matrix is positive definite, positive semi-definite, negative
        semi-definite, negative definite of indefinite, respectively
        If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix.T, matrix):
        return None
    eigen_values, _ = np.linalg.eig(matrix)
    if all([value > 0 for value in eigen_values]):
        return "Positive definite"
    if all([value >= 0 for value in eigen_values]):
        return "Positive semi-definite"
    if all([value < 0 for value in eigen_values]):
        return "Negative definite"
    if all([value <= 0 for value in eigen_values]):
        return "Negative semi-definite"
    else:
        return "Indefinite"
