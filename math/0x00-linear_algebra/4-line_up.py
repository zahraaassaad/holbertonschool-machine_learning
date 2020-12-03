#!/usr/bin/env python3

"""
Module for add_arrays.
"""


def add_arrays(arr1, arr2):
    """Adds 2 arrays."""
    if len(arr1) != len(arr2):
        return None
    sum = []
    for elem in range(len(arr1)):
        sum.append(arr1[elem] + arr2[elem])
    return sum
