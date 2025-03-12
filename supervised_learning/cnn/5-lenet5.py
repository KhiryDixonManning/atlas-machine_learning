#!/usr/bin/env python3
""" module containing function that builds a modified version of the
    LeNet-5 architecture using Keras."""
import tensorflow.keras as K


def lenet5(X):
    """ function that builds a modified version of the LeNet-5 architecture
        using Keras.

    PARAMETERS
    ==========
        X [K.Input]: Input images for the networkof Shape (m, 28, 28, 1)
            m - number of images

    RETURNS
    =======
        keras model
    """
    model = K.Sequential()
    model.add(K.layers.Conv2D(kernel_initializer=K.initializers.HeNormal(),
                              filters=6, kernel_size=(5, 5), padding='same',
                              activation='relu', input_shape=X.shape[1:]))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Conv2D(kernel_initializer=K.initializers.HeNormal(),
                              filters=16, kernel_size=(5, 5), padding='valid',
                              activation='relu'))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(kernel_initializer=K.initializers.HeNormal(),
                             units=120, activation='relu'))
    model.add(K.layers.Dense(kernel_initializer=K.initializers.HeNormal(),
                             units=84, activation='relu'))
    model.add(K.layers.Dense(kernel_initializer=K.initializers.HeNormal(),
                             units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
