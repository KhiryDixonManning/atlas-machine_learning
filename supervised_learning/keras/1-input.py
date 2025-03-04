#!/usr/bin/env python3
""" module containing function that builds a neural network with
    the Keras library """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ function that builds a neural network with the Keras library

        PARAMETERS
        ==========
            nx [int]: number of input features to the network
            layers [list]: contains the number of nodes in each layer of
                        the network
            activations [list]: contains the activation functions used for
                            each layer of the network
            lambtha [floay]: L2 regularization parameter
            keep_prob [float]: probability that a node will be kept for dropout

        RETURNS
        =======
            the model
    """
    L2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx, ))
    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(inputs)
        elif i == len(layers) - 1:
            drop = K.layers.Dropout(1 - keep_prob)(x)
            outputs = K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2)(drop)
        else:
            drop = K.layers.Dropout(1 - keep_prob)(x)
            x = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=L2)(drop)
    return K.Model(inputs, outputs)
