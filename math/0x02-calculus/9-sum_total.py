#!/usr/bin/env python3
"""Calculates the sum of variable squared."""


def summation_i_squared(n):
        """Module for summation of n squared."""
            if type(n) != int or n < 1:
                        return None
            else:
                        return int((n * (n + 1) * (2 * n + 1)) / 6)
