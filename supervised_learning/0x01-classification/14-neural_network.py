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

    def forward_prop(self, X):
        """Calculates the forward propagation."""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model."""
        cost_array = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neurons predictions."""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        return (np.where(self.__A2 > 0.5, 1, 0), cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent."""
        dz2 = A2 - Y
        dw2 = (1 / len(Y[0])) * np.matmul(dz2, A1.T)
        db2 = (1 / len(Y[0])) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / len(Y[0])) * np.matmul(dz1, X.T)
        db1 = (1 / len(Y[0])) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000,
              alpha=0.05):
        """Trains the neuron."""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if i != iterations:
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return (A, cost)
