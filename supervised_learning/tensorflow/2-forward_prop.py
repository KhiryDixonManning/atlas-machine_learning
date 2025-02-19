#!/usr/bin/env python3
""" module containing function that creates a forward propagation graph
    for the NN """
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ function that creates a forward propagation graph for the NN

        Parameters:
            x [symbtensor] - placeholder for the input data
            layer_sizes [list] - list containing the number of nodes in
                                    each layer of the network
            activations [list] - list containing the activation functions for
                                    each layer of the network

        Returns:
            [tensor] - the prediction of the network in tensor form
        """
    create_layer = __import__('1-create_layer').create_layer
    for lay in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[lay], activations[lay])
    return x
