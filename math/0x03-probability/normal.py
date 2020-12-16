#!/usr/bin/env python3
"""
normal.py
Module defining normal distribution
"""


class Normal:
    """ Normal class. It defines normal distribution.
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = (sum(data) / len(data))
                s = 0
                for i in range(0, len(data)):
                    s = s + ((data[i] - self.mean))**2
                self.stddev = (s/len(data))**(1/2)

    def z_score(self, x):
        """Calculates the z-score."""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Calculates the x-score."""
        return self.stddev * z + self.mean
