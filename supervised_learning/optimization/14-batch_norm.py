#!/usr/bin/env python3
"""Module containing a function that creates a batch normalization layer
   for a neural network in TensorFlow."""

import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
        prev (tensor): Activated output of the previous layer.
        n (int): Number of nodes in the layer to be created.
        activation (function): Activation function to be applied to the output
                               of the layer.

    Returns:
        tensor: Activated output for the batch normalization layer.
    """
    # Initialize weights using VarianceScaling initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create a dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        name="batch_norm_layer",
        kernel_initializer=init
    )

    # Apply the dense layer to the previous layer
    x = layer(prev)

    # Compute the mean and variance for batch normalization
    mean, variance = tf.nn.moments(x, axes=[0])

    # Create variables for scale (gamma) and offset (beta)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    # Apply batch normalization
    norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)

    # Apply the activation function
    activated = tf.keras.layers.Activation(activation)

    return activated(norm)
