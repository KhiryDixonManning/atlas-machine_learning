#!/usr/bin/env python3
""" module containing function that creates a layer of a neural network using
    dropout """
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ function that creates a layer of a neural network using dropout


        PARAMETERS
        ==========
        prev [tensor]: the output of the previous layer

        n [int]: number of nodes the new layer should contain

        activation [string]: activation function that should be used on the
                                layer

        keep_prob [?]: probability that a node will be kept


        RETURNS
        =======
        [?]: output of new layer
    """
    tf.layers.Dropout(keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=init
    )
    return layer(prev)
