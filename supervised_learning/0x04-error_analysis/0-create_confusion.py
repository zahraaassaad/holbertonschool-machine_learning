#!/usr/bin/env python3

"""creates a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Returns: a confusion numpy.ndarray."""
    return np.matmul(labels.T, logits)
