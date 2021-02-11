#!/usr/bin/env python3
"""builds an inception block as
   described in Going Deeper with Convolutions (2014)"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Returns: the concatenated output
       of the inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal")(A_prev)
    conv3r = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             activation='relu', padding="same",
                             kernel_initializer="he_normal")(A_prev)
    conv3 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal")(conv3r)
    conv5r = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             activation='relu', padding="same",
                             kernel_initializer="he_normal")(A_prev)
    conv5 = K.layers.Conv2D(filters=F5,
                            kernel_size=(5, 5),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal")(conv5r)
    pool3 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(1, 1), padding="same")(A_prev)
    convf = K.layers.Conv2D(filters=FPP,
                            kernel_size=(1, 1),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal")(pool3)
    incept_block = K.layers.Concatenate(axis=3)([conv1, conv3, conv5, convf])
    return incept_block
