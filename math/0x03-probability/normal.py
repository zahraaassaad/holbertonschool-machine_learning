#!/usr/bin/env python3
"""
normal.py
Module defining normal distribution
"""


class Normal:
    """ Normal class. It defines normal distribution.
    """

    PI = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the data."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            s = 0
            for i in range(0, len(data)):
                s += ((data[i] - self.mean))**2
            self.stddev = (s/len(data))**(1/2) 