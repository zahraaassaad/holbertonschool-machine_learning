#!/usr/bin/env python3

"""creates a layer.
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer."""
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(units=n, activation=activation,
                            kernel_initializer=w, name="layer")
    y = layer(prev)
    return (y)
