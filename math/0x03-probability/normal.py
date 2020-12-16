#!/usr/bin/env python3
"""
normal.py
Module defining normal distribution
"""


class Normal:
    """ Normal class. It defines normal distribution.
    """

    PI = 3.1415926536
    
    @classmethod
    def get_erf(cls, x):
        seq = x - ((x ** 3)/3) + ((x ** 5)/10) - ((x ** 7)/42) + ((x ** 9)/216)
        erf = (2/(cls.PI ** (1/2))) * seq
        return erf
                                
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

    def z_score(self, x):
        """Calculates the z-score."""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """Calculates the x-score."""
        return self.stddev * z + self.mean
    
    def pdf(self, x):
        """Calculates the value of the PDF."""
        return (2.7182818285**((-1/2) * (((
            x - self.mean) / self.stddev)**2))) * (
            1 / (self.stddev * (2 * self.PI)**(1/2)))

    def cdf(self, x):
        """Calculates the value of the CDF."""
        expression = (x - self.mean)/(self.stddev * (2 ** (1/2)))
        return (1/2) * (1 + self.get_erf(expression))
