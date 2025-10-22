#!/usr/bin/env python3
"""" module containing function that builds the ResNet-50 architecture """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ function that builds the ResNet-50 architecture

    RETURNS
    =======
        [keras.Model]: The constructed ResNet-50 model.
    """
    inp = K.layers.Input((224, 224, 3))
    init = K.initializers.HeNormal()
    x1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                         kernel_initializer=init,
                         padding='same')(inp)
    x1 = K.layers.BatchNormalization(axis=3)(x1)
    x1 = K.layers.Activation('relu')(x1)
    x1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                               padding='same')(x1)

    x2 = projection_block(x1, [64, 64, 256], 1)
    x2 = identity_block(x2, [64, 64, 256])
    x2 = identity_block(x2, [64, 64, 256])

    x3 = projection_block(x2, [128, 128, 512], 2)
    x3 = identity_block(x3, [128, 128, 512])
    x3 = identity_block(x3, [128, 128, 512])
    x3 = identity_block(x3, [128, 128, 512])

    x4 = projection_block(x3, [256, 256, 1024], 2)
    x4 = identity_block(x4, [256, 256, 1024])
    x4 = identity_block(x4, [256, 256, 1024])
    x4 = identity_block(x4, [256, 256, 1024])
    x4 = identity_block(x4, [256, 256, 1024])
    x4 = identity_block(x4, [256, 256, 1024])

    x5 = projection_block(x4, [512, 512, 2048], 2)
    x5 = identity_block(x5, [512, 512, 2048])
    x5 = identity_block(x5, [512, 512, 2048])

    x6 = K.layers.AveragePooling2D((7, 7), padding='same')(x5)

    x7 = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=init)(x6)
    return K.Model(inp, x7)
