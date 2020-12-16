#!/usr/bin/env python3
"""
binomial.py
Module defining binomial distribution
"""


class Binomial:
    """Binomial class. It defines binomial distribution.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the data."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p >= 1 or p <= 0:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            else:
                sum_x = 0
                ux = sum(data) / len(data)
                for i in range(0, len(data)):
                    sum_x = sum_x + (data[i] - ux)**2
                var_x = sum_x / len(data)
                P = 1 - (var_x / ux)
                self.n = int(round(ux / P))
                self.p = float(ux / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF."""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        factorial_k = 1
        factorial_n = 1
        n = self.n
        factorial_n_k = 1
        if k != 0:
            for i in range(1, k + 1):
                factorial_k = factorial_k * i
        for j in range(1, n + 1):
            factorial_n = factorial_n * j
        for l in range(1, n - k + 1):
            factorial_n_k = factorial_n_k * l
        return (factorial_n / (factorial_k * factorial_n_k)) * (
            self.p**k) * ((1 - self.p)**(self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF."""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        CDF = 0
        for i in range(0, k+1):
            CDF = CDF + self.pmf(i)
        return CDF
