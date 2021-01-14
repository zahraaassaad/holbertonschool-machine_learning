#!/usr/bin/env python3
"""updates variable using gradient
   descent with momentum"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """return updated variable and gradient"""
    Vdw = beta1 * v + (1 - beta1) * grad
    W = var - alpha * Vdw
    return W, Vdw
