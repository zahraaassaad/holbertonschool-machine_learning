#!/usr/bin/env python3
"""creates training operation
   using gradient descent with momentum
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """tensor with operation"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)
    return train
