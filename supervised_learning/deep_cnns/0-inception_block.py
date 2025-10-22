#!/usr/bin/env python3
"""" module containing function that builds an inception block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ function that builds an inception block

    PARAMETERS
    ==========
        A_prev [tensor]: Output from the previous layer.
        filters [tuple or list]: A tuple or list containing the following
                                    filter sizes:
            F1 [int]: # of filters in the 1x1 convolution.
            F3R [int]: # of filters in the 1x1 convolution before the 3x3
                        convolution.
            F3 [int]: # of filters in the 3x3 convolution.
            F5R [int]: # of filters in the 1x1 convolution before the 5x5
                        convolution.
            F5 [int]: # of filters in the 5x5 convolution.
            FPP [int]: # of filters in the 1x1 convolution after the
                        max pooling.

    RETURNS
    =======
        [tensor]: The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    L1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                         activation='relu')(A_prev)

    L2 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                         activation='relu')(A_prev)
    L2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                         activation='relu')(L2)

    L3 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                         activation='relu')(A_prev)
    L3 = K.layers.Conv2D(F5, (5, 5), padding='same',
                         activation='relu')(L3)

    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                           activation='relu')(pool)
    return K.layers.concatenate([L1, L2, L3, pool], axis=3)
