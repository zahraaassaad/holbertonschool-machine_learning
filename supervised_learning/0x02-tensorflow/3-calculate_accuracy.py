#!/usr/bin/env python3

"""calculates the accuracy of a prediction.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Returns: a tensor containing the decimal accuracy.
    """
    equal = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return accuracy
