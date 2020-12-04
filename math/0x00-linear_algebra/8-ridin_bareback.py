#!/usr/bin/env python3

"""
Module for mat_mul.
"""


def mat_mul(mat1, mat2):
    """Multiply 2 arrays."""
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = [[sum(a * b for a, b in zip(mat1_row, mat2_col))
                for mat2_col in zip(*mat2)]
               for mat1_row in mat1]
    return new_mat
