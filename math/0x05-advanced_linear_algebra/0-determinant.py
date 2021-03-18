#!/usr/bin/env python3

"""
0-determinant.py
calculates the determinant of a matrix
"""

from numpy import linalg as LA


def determinant(matrix):
    """
    Returns: the determinant of matrix
    """
    if matrix == [[]]:
        return 1
    if len(matrix) == 0 or type(matrix) != list or type(matrix[0]) != list:
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) != len(matrix):
        raise ValueError("matrix must be a square matrix")
    return int(LA.det(matrix))
