#!/usr/bin/env python3

"""
Module for matrix_shape.
"""


def matrix_shape(matrix):
    """Finds the shape of matrix."""
    dims = [len(matrix)]
    while type(matrix[0]) != int:
        dims.append(len(matrix[0]))
        matrix = matrix[0]
    return dims
