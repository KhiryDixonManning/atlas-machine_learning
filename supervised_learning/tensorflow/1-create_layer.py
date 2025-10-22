#!/usr/bin/env python3
"""Module containing a function that creates a layer for a TensorFlow neural network."""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function that creates a layer for a TensorFlow neural network.

    Parameters:
        prev (tensor): The output tensor from the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (function): The activation function to be used by the layer.

    Returns:
        tensor: The output tensor for the current layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=init
    )
    return layer(prev)
