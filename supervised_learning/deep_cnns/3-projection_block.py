#!/usr/bin/env python3
"""" module containing function that builds a projection block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ function that builds a projection block

    PARAMETERS
    ==========
        A_prev [tensor]: Output from the previous layer.
        filters [tuple or list]: A tuple or list containing the following
                                    filter sizes:
            F11 [int]: Number of filters in the first 1x1 convolution.
            F3 [int]: Number of filters in the 3x3 convolution.
            F12 [int]: Number of filters in the second 1x1 convolution
                        as well as the 1x1 convolution in the shortcut
                        connection.
        s [int]: The stride of the first convolution in both the main path
                    and the shortcut connection. Defaults to 2.

    RETURNS
    =======
        [tensor]: The activated output of the projection block.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal()

    x1 = K.layers.Conv2D(F11, (1, 1), kernel_initializer=init,
                         padding='same', strides=(s, s))(A_prev)
    x1 = K.layers.BatchNormalization(axis=3)(x1)
    x1 = K.layers.Activation('relu')(x1)

    x2 = K.layers.Conv2D(F3, (3, 3), kernel_initializer=init,
                         padding='same', strides=(1, 1))(x1)
    x2 = K.layers.BatchNormalization(axis=3)(x2)
    x2 = K.layers.Activation('relu')(x2)

    x3 = K.layers.Conv2D(F12, (1, 1), kernel_initializer=init,
                         padding='same', strides=(1, 1))(x2)
    x3 = K.layers.BatchNormalization(axis=3)(x3)

    x_sc = K.layers.Conv2D(F12, (1, 1), kernel_initializer=init,
                           padding='same', strides=(s, s))(A_prev)
    x_sc = K.layers.BatchNormalization(axis=3)(x_sc)

    x = K.layers.Add()([x3, x_sc])
    return K.layers.Activation('relu')(x)
