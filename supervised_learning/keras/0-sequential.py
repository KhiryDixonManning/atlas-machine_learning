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
    model = K.models.Sequential()
    L2 = K.regularizers.L2(lambtha)
    model.add(K.layers.Dense(layers[0], input_shape=(nx, ),
                             activation=activations[0], kernel_regularizer=L2))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=L2))
    return model
