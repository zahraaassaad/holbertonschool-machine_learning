#!/usr/bin/env python3
"""determines if you should stop
   gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """a boolean of whether the network should be stopped
       early, followed by the updated count"""
    if (opt_cost - cost) <= threshold:
        if count + 1 >= patience:
            return True, count + 1
        else:
            return False, count + 1
    else:
        return False, 0
