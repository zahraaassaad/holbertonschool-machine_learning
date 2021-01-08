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

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions."""
        self.forward_prop(X)
        cost = self.cost(Y, self._A)
        return (np.where(self._A > 0.5, 1, 0), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron."""
        cost = self.cost(Y, A)
        dz = A - Y
        dw = (1 / len(Y[0])) * np.matmul(dz, X.T)
        db = (1 / len(Y[0])) * np.sum(dz)
        self.__W -= np.multiply(alpha, dw)
        self.__b -= np.multiply(alpha, dz)