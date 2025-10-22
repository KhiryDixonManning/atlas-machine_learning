#!/usr/bin/env python3
"""Module containing a function that creates the training operation for a
   neural network in TensorFlow using the Adam optimization algorithm."""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in TensorFlow
       using the Adam optimization algorithm.

    Parameters:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.

    Returns:
        train (tensor operation): The Adam optimization operation.
    """
    optim = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                   beta2=beta2, epsilon=epsilon)
    train = optim.minimize(loss)
    return train
