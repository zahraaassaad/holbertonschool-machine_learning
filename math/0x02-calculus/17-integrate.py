#!/usr/bin/env python3
"""Calculates the integral of a function."""


def poly_integral(poly, C=0):
    """Module for integral."""
    new_L = list(range(len(poly)+1))
    i = 1

    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    new_L[0] = C
    if sum(poly) == 0:
        return new_L
    if len(poly) == 1:
        new_L[1] = poly[0]
        return new_L
    while i < len(new_L):
        if poly[i - 1] % i == 0:
            new_L[i] = int(poly[i - 1] / (i))
        else:
            new_L[i] = poly[i - 1] / (i)
        i += 1
    return new_L
