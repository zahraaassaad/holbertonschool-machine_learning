#!/usr/bin/env python3

"""creates a tensorflow layer
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Returns the output of the new layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_regu = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=w, name="layer",
                            kernel_regularizer=l2_regu)
    y = layer(prev)
    return (y)
