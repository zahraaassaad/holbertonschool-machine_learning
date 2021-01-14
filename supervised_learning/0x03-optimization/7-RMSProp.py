#!/usr/bin/env python3
"""updates variable using RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """return updated variable and gradient"""
    Sdw = beta2 * s + (1 - beta2) * (grad**2)
    W = var - alpha * grad / (Sdw ** (1/2) + epsilon)
    return W, Sdw
