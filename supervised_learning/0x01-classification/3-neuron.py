#!/usr/bin/env python3
"""
0-neuron.py
Module defining a neuron
"""
import numpy as np


class Neuron:
    """Neuron class. Defines a neuron.
    """
    def __init__(self, nx):
        """Initializes the data."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for A."""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron."""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model."""
        array = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1.0000001 - A))
        cost = - np.sum(array) / len(A[0])
        return cost
