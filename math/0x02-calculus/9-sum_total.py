#!/usr/bin/env python3
"""Calculates sum of variable squared"""


def summation_i_squared(n):
    """Module for summation i squared."""
    if type(n) != int:
        return None
    if n < 1:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
                                
