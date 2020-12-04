#!/usr/bin/env python3

"""
Module for add_matrices2D.
"""


def add_matrices2D(mat1, mat2):
    """Adds 2 arrays."""
    if len(mat1[0]) != len(mat2[0]):
        return None
    new_mat = []
    for row in range(len(mat1)):
        trans = []
        for col in range(len(mat1[0])):
            trans.append(mat1[row][col] + mat2[row][col])
        new_mat.append(trans)
    return new_mat
