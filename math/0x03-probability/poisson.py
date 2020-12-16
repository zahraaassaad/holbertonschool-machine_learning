#!/usr/bin/env python3
"""
poisson.py
Module defining poisson distribution
"""


class Poisson:
    """ Poisson class. It defines poisson distribution.
    """
    def __init__(self, data=None, lambtha=1.):
        """Initializes the data."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF."""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        k_factorial = 1
        if k != 0:
            for i in range(2, k+1):
                k_factorial = k_factorial * i
        return ((self.lambtha ** k) *
                (2.7182818285 ** (-(self.lambtha)))) / k_factorial
