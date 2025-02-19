#!/usr/bin/env python3
""" module containing function that calculates the softmax cross-entropy
    loss of a prediction """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ function that calculates the softmax cross-entropy loss of a
        prediction

        Parameters:
            y [symbtensor] - placeholder for the labels of the input data
            y_pred [tensor] - tensor containing the network’s predictions

        Returns:
            [tensor] - tensor containing the loss of the prediction
        """
    return tf.compat.v1.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )