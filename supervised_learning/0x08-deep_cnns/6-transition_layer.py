#!/usr/bin/env python3
"""builds a transition layer
   as described in Densely Connected
   Convolutional Networks"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Returns: The output of the
       transition layer and the number
       of filters within the output, respectively"""
    BN1 = K.layers.BatchNormalization(axis=3)(X)
    Relu1 = K.layers.Activation("relu")(BN1)
    conv1 = K.layers.Conv2D(int(compression * nb_filters),
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(Relu1)
    pool5 = K.layers.AveragePooling2D(pool_size=(2, 2),
                                      strides=(2, 2))(conv1)
    return pool5, int(compression * nb_filters)
