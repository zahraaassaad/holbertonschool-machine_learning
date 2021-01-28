#!/usr/bin/env python3
"""save and load a model using Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """loads an entire model"""
    return K.models.load_model(filename)
