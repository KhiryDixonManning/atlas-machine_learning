#!/usr/bin/env python3
"""" module containing function that builds an inception network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network

    RETURNS
    =======
        [keras.Model] The constructed inception network.
    """
    init = K.initializers.HeNormal()
    inp = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, (7, 7), kernel_initializer=init,
                            padding='same', strides=(2, 2),
                            activation='relu')(inp)
    pool1 = K.layers.MaxPooling2D((3, 3), padding='same',
                                  strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(64, (1, 1), kernel_initializer=init,
                            padding='same', strides=(1, 1),
                            activation='relu')(pool1)
    conv22 = K.layers.Conv2D(192, (3, 3), kernel_initializer=init,
                             padding='same', strides=(1, 1),
                             activation='relu')(conv2)
    pool2 = K.layers.MaxPooling2D((3, 3), padding='same',
                                  strides=(2, 2))(conv22)
    block1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    block2 = inception_block(block1, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D((3, 3), padding='same',
                                  strides=(2, 2))(block2)
    block3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    block4 = inception_block(block3, [160, 112, 224, 24, 64, 64])
    block5 = inception_block(block4, [128, 128, 256, 24, 64, 64])
    block6 = inception_block(block5, [112, 144, 288, 32, 64, 64])
    block7 = inception_block(block6, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D((3, 3), padding='same',
                                  strides=(2, 2))(block7)
    block8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    block7 = inception_block(block8, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AvgPool2D((7, 7), padding='valid',
                               strides=(1, 1))(block7)
    dropped = K.layers.Dropout(0.4)(pool5)
    dense = K.layers.Dense(1000, kernel_initializer=init,
                           activation='softmax')(dropped)
    model = K.Model(inp, dense)
    return model
