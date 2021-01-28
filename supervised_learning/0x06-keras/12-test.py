#!/usr/bin/env python3
"""tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Returns: the loss and accuracy of the model"""
    return network.evaluate(data, labels, verbose=verbose)
