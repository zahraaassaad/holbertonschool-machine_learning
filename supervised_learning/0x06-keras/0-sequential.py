#!/usr/bin/env python3

"""builds a neural network with the Keras library
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Returns: the keras model"""
    model = k.Sequential()
    regularization = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=regularization, input_dim=nx))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=regularization))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
