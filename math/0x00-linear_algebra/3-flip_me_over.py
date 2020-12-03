#!/usr/bin/env python3

"""
Module for matrix_transpose.
"""


def matrix_transpose(matrix):
    """Transposes a matrix."""
    new_mat = []
    for col in range(len(matrix[0])):
        trans = []
        for row in range(len(matrix)):
            trans.append(matrix[row][col])
        new_mat.append(trans)
    return new_mat
