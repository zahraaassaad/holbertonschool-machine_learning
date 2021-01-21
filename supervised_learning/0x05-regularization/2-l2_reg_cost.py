#!/usr/bin/env python3
"""calculates the cost """
import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost"""
    regularization_losses = tf.losses.get_regularization_losses()
    return cost + regularization_losses
