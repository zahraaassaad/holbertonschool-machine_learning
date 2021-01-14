#!/usr/bin/env python3
"""creates training operation
   using Adam optimization
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """returns tensor with Adam optimization"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train = optimizer.minimize(loss)
    return train
