#!/usr/bin/env python3
"""Module containing a function that calculates the softmax cross-entropy
   loss of a prediction."""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Function that calculates the softmax cross-entropy loss of a prediction.

    Parameters:
        y (symbtensor): Placeholder for the labels of the input data.
        y_pred (tensor): Tensor containing the network's predictions.

    Returns:
        tensor: A tensor containing the loss of the prediction.
    """
    return tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )
