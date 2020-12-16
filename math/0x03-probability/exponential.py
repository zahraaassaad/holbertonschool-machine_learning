#!/usr/bin/env python3
"""
exponential.py
Module defining exponential distribution
"""


class Exponential:
    """ Exponential class. It defines poisson distribution.
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
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """Calculates the value of the PDF."""
        if x < 0:
            return 0
        return ((self.lambtha) *
                (2.7182818285 ** (-(self.lambtha) * x)))

    def cdf(self, x):
        """Calculates the value of the CDF."""
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-(self.lambtha * x)))
