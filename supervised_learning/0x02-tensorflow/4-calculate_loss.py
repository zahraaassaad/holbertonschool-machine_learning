#!/usr/bin/env python3

"""
calculates the softmax cross-entropy loss.
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Returns: a tensor containing the loss.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
