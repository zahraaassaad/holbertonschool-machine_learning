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

    def forward_prop(self, X):
        """Calculates the forward propagation."""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w],
                          self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            self.__cache[a_new] = 1 / (1 + np.exp(-Z))
        Act = "A" + str(self.__L)
        return (self.__cache[Act], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model."""
        cost_array = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.where(A > 0.5, 1, 0), cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent."""
        weights_copy = self.__weights.copy()
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i - 1)]
            dw = (1 / len(Y[0])) * np.matmul(dz, A.T)
            db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
            w = "W" + str(i)
            b = "b" + str(i)
            self.__weights[w] = self.__weights[w] - alpha * dw
            self.__weights[b] = self.__weights[b] - alpha * db
            dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (A * (1 - A))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the deep neural network."""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        _, cache = self.forward_prop(X)
        for i in range(iterations):
            self.gradient_descent(Y, cache, alpha)
            _, cache = self.forward_prop(X)
        return self.evaluate(X, Y)
