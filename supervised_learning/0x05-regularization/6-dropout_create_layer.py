#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Returns: the output of the new layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    Drop_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=w, name="layer",
                            kernel_regularizer=Drop_reg)
    y = layer(prev)
    return (y)
