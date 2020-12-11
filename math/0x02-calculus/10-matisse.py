#!/usr/bin/env python3
"""Calculates the derivative of a function."""


def poly_derivative(poly):
    """Module for derivative."""
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    if poly is None:
        return None
    new_L = list(range(len(poly)-1))
    for i in range(len(new_L)):
        if type(i) != int:
            return None
        new_L[i] = poly[i + 1] * (i + 1)
    return new_L
