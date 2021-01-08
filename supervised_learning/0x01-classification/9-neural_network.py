#!/usr/bin/env python3
"""
8-neural_network.py
Module defining a neural network
"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork class. Defines a neural network.
    """
    def __init__(self, nx, nodes):
        """Initializes the data."""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

        @property
        def W1(self):
            """Getter for weights1."""
            return self.__W1

        @property
        def b1(self):
            """Getter for bias1."""
            return self.__b1

        @property
        def A1(self):
            """Getter for A1."""
            return self.__A1

        @property
        def W2(self):
            """Getter for weights2."""
            return self.__W2

        @property
        def b2(self):
            """Getter for bias2."""
            return self.__b2

        @property
        def A2(self):
            """Getter for A2."""
            return self.__A2
