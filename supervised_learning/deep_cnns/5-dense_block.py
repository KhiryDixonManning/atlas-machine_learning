#!/usr/bin/env python3
"""" module containing function that builds a dense block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ Builds a dense block

    Parameters:
        X [tensor]: Output from the previous layer.
        nb_filters [int]: Number of filters in X.
        growth_rate [int]: Growth rate for the dense block.
        layers [int]: Number of layers in the dense block.

    Returns:
        [tuple]: The concatenated output of each layer within the Dense Block
                    and the number of filters within the concatenated outputs,
                    respectively.
    """
    out = X
    for _ in range(layers):
        init = K.initializers.HeNormal()
        x1 = K.layers.BatchNormalization(axis=3)(out)
        x1 = K.layers.Activation('relu')(x1)
        x1 = K.layers.Conv2D(growth_rate*4, (1, 1),
                             kernel_initializer=init,
                             padding='same')(x1)
        x2 = K.layers.BatchNormalization(axis=3)(x1)
        x2 = K.layers.Activation('relu')(x2)
        x2 = K.layers.Conv2D(growth_rate, (3, 3),
                             kernel_initializer=init,
                             padding='same')(x2)
        out = K.layers.concatenate([out, x2])
        nb_filters = nb_filters + growth_rate
    return out, nb_filters
