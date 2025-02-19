#!/usr/bin/env python3
"""Module containing a function that creates a forward propagation graph
   for the neural network."""

import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates a forward propagation graph for a neural network.

    Parameters:
        x (symbtensor): Placeholder for the input data.
        layer_sizes (list): List containing the number of nodes in each layer
                             of the network.
        activations (list): List containing the activation functions for each
                             layer of the network.

    Returns:
        tensor: The prediction of the network in tensor form.
    """
    create_layer = __import__('1-create_layer').create_layer
    for lay in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[lay], activations[lay])
    return x
