#!/usr/bin/env python3
"""updates learning rate
   using inverse time decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """"returns updated value for alpha"""
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
