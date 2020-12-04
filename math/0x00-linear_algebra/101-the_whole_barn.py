#!/usr/bin/env python3

"""
Module for add_matrices.
"""


def matrix_shape(matrix):
    """Finds shape of matrix."""
    dimensions = []
    len_matrix = len(matrix)
    dimensions.append(len_matrix)
    while type(matrix[0]) == list:
        matrix = matrix[0]
        len_matrix = len(matrix)
        dimensions.append(len_matrix)
    return dimensions


def add_arrays(arr1, arr2):
    """Adds arrays."""
    addition = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        addition.append(arr1[i] + arr2[i])
    return addition


def add_matrices2D(mat1, mat2):
    """Adds 2D matrices"""
    addition = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    m = [[mat1[i][j] + mat2[i][j] for j in range(len(
        mat2[0]))] for i in range(len(mat2))]
    return m


def add_matrices(mat1, mat2):
    """Adds matrices."""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    if type(mat1[0]) != list:
        return add_arrays(mat1, mat2)
    if len(shape1) == 2:
        return add_matrices2D(mat1, mat2)
    added_matrix = []
    for i in range(len(mat1)):
        added_matrix.append(add_matrices(mat1[i], mat2[i]))
    return added_matrix
