#!/usr/bin/env python3
"""Calculates the integral of a function."""


def poly_integral(poly, C=0):
    """Module for integral."""
    inte = []
    i = 0

    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    inte.append(C)
    if sum(poly) == 0:
        return inte
    if len(poly) == 1:
        inte.append(poly[0])
        return inte
    while i < len(poly):
        if poly[i] % (i + 1) == 0:
            inte.append(int(poly[i]/(i + 1)))
        else:
            inte.append(poly[i]/(i + 1))
        i += 1
    return inte
