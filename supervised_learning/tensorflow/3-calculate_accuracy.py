#!/usr/bin/env python3
"""Module containing a function that calculates the accuracy of predictions."""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a TensorFlow model's predictions.

    Parameters:
        y (symbtensor): Placeholder for the labels of the input data.
        y_pred (tensor): Tensor containing the network's predictions.

    Returns:
        tensor: A tensor containing the decimal accuracy of the prediction.
    """
    pred = tf.math.argmax(y_pred, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))
