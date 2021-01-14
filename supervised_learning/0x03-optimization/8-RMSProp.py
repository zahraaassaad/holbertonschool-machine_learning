#!/usr/bin/env python3
"""creates training operation
   using gradient descent with RMSProp
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """returns tensor with RMSProp"""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    train = optimizer.minimize(loss)
    return train
