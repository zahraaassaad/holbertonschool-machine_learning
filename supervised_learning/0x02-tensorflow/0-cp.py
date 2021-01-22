#!/usr/bin/env python3

"""Returns two placeholders.
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Returns: placeholders named x and y."""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return (x, y)
