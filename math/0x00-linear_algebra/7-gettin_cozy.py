#!/usr/bin/env python3

"""
Module for cat_matrices2D.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatinates 2 matrices."""
    new_mat = []
    if len(mat1) != len(mat2[0]) and len(mat1[0]) != len(mat2):
        return None
    if axis == 0:
        for item in mat1:
            new_mat.append(item[:])
        for row in mat2:
            new_mat.append(row[:])
        return new_mat
    for row in range(len(mat1)):
        new_mat.append(mat1[row] + mat2[row])
    return new_mat
