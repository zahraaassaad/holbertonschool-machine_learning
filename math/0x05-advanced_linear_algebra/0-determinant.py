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
    ind = list(range(len(matrix)))
    total = 0
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix[0]) == 0:
        return 1
    if len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix[0]) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    for i in ind:
        mat_copy = matrix
        mat_copy = mat_copy[1:]
        height = len(mat_copy)
        for j in range(height):
            mat_copy[j] = mat_copy[j][:i] + mat_copy[j][i+1:]
    return total
