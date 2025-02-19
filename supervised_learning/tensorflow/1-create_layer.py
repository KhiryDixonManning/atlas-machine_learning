#!/usr/bin/env python3
""" module containing function that creates a layer of the tensorflow NN """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ function that creates a layer of the tensorflow NN

        Parameters:
            prev [tensor] - the tensor output of the previous layer
            n [int] - the number of nodes in the layer to create
            activation [?] - the activation function that the layer should use

        Returns:
            [tensor] tensor output for the layer
        """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=init
    )
    return layer(prev)
