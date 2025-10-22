#!/usr/bin/env python3
"""Module containing a function that creates the training operation for a
   neural network in TensorFlow using the gradient descent with momentum
   optimization algorithm."""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network in TensorFlow
       using the gradient descent with momentum optimization algorithm.

    Parameters:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        tf.Operation: The momentum optimization operation.
    """
    # Initialize the MomentumOptimizer with given learning rate and momentum
    optim = tf.train.MomentumOptimizer(alpha, beta1)

    # Create the training operation using the optimizer and loss
    train = optim.minimize(loss)

    return train
