#!/usr/bin/env python3
"""" module containing function that builds the transition layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer

    Parameters:
        X [tensor]: Output from the previous layer.
        nb_filters [int]: Number of filters in X.
        compression [float]: Compression factor for the transition layer.

    Returns:
        [tuple]: The output tensor from the transition layer and the number of
                filters within the output tensor.
    """
    nb_filters = int(nb_filters * compression)
    init = K.initializers.HeNormal()
    x1 = K.layers.BatchNormalization(axis=3)(X)
    x1 = K.layers.Activation('relu')(x1)
    x1 = K.layers.Conv2D(nb_filters, (1, 1),
                         kernel_initializer=init,
                         padding='same')(x1)
    x1 = K.layers.AveragePooling2D((2, 2))(x1)
    return x1, nb_filters
