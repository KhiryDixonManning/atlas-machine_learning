#!/usr/bin/env python3
""" module containing function that creates a tensorflow layer that includes L2
    regularization """
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ function that creates a tensorflow layer that includes L2
        regularization


        PARAMETERS
        ==========
        prev [tensor]: the output of the previous layer

        n [int]: number of nodes the new layer should contain

        activation [string]: activation function that should be used on the
                                layer

        lambtha [float]: L2 regularization parameter


        RETURNS
        =======
        []: output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=init,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
