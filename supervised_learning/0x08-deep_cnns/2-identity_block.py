#!/usr/bin/env python3
"""builds an identity block as described in Deep Residual
   Learning for Image Recognition (2015)"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Returns: the activated output of
       the identity block"""
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(A_prev)
    BN1 = K.layers.BatchNormalization(axis=3)(conv1)
    Relu1 = K.layers.Activation("relu")(BN1)
    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(Relu1)
    BN2 = K.layers.BatchNormalization(axis=3)(conv2)
    Relu2 = K.layers.Activation("relu")(BN2)
    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(Relu2)
    BN3 = K.layers.BatchNormalization(axis=3)(conv3)
    add_layers = K.layers.Add()([BN3, A_prev])
    return K.layers.Activation("relu")(add_layers)
