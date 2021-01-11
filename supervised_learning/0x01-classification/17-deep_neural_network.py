#!/usr/bin/env python3
"""
16-deep_neural_network.py
Module defining a deep neural network
"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class. Defines a deep neural network.
    """
    def __init__(self, nx, layers):
        """Initializes the data."""
        if type(nx) != int:
            raise TypeError("nx must be an integers")
        if nx < 1:
            raise ValueError("nx must be a positive integers")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer = 1
        layer_size = nx
        for i in layers:
            if type(i) != int or i <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(layer)
            b = "b" + str(layer)
            self.__weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.__weights[b] = np.zeros((i, 1))
            layer += 1
            layer_size = i

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights
