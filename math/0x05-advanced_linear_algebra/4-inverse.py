#!/usr/bin/env python3

"""
4-inverse.py
calculates the inverse of a matrix
"""


def inverse(matrix):
    """Returns: the inverse of matrix, or None if matrix is singular"""
    deter_mat = determinant(matrix)
    adj_mat = adjugate(matrix)
    if deter_mat is 0:
        return None
    inv_mat = []
    for i in range(len(matrix)):
        row = [adj_mat[i][j]/deter_mat for j in range(len(matrix[i]))]
        inv_mat.append(row)
    return inv_mat


def adjugate(matrix):
    """ FUnction Adjugate"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if not all([len(row) == len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    matrix_cof = cofactor(matrix)
    return [[row[i] for row in matrix_cof] for i in range(len(matrix_cof[0]))]


def cofactor(matrix):
    """
    Returns: the cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    result_cofactor = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result_cofactor[i][j] = (-1) ** (i + j) * result_cofactor[i][j]
    return result_cofactor


def minor(matrix):
    """
    Returns: the minor of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if not all([len(row) == len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) is 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    result = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            minor_row.append(determinant(getMatrixMinor(matrix, i, j)))
        result.append(minor_row)
    return result


def getMatrixMinor(m, i, j):
    """calculates the minor of a squared matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]


def determinant(matrix):
    """
    Returns: the determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    square = [len(row) == len(matrix) for row in matrix]
    if not all(square):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) is 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    deter = 0
    for c in range(len(matrix)):
        deter += ((-1) ** c) * matrix[0][c] *\
                 determinant(getMatrixMinor(matrix, 0, c))
    return deter
