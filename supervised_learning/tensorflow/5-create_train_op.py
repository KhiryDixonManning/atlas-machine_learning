#!/usr/bin/env python3
"""Module containing a function that creates the training operation for
   the network."""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network.

    Parameters:
        loss (float): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        operation: An operation that trains the network using gradient descent.
    """
    optim = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train = optim.minimize(loss)
    return train
