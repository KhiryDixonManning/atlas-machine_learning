#!/usr/bin/env python3
""" module containing function that conducts forward propagation using
    Dropout """
import numpy as np


def softmax(z):
    """ softmax function """
    ez = np.exp(z - np.max(z))
    return ez / np.sum(ez, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """ function that conducts forward propagation using Dropout


        PARAMETERS
        ==========
        X [np.ndarray]: array of shape (nx, m) containing the input data
                        for the network
                        > nx [int]: number of input features
                        > m [int]: number of data points

        weights [dict]: the weights and biases of the neural network

        L [int]: number of layers in the network

        keep_prob [?]: probability that a node will be kept


        RETURNS
        =======
        [dict]: the outputs of each layer and the dropout mask used on
                each layer
    """
    cache = {}
    cache["A0"] = X
    A = X
    for lay in range(1, L + 1):
        W = weights["W" + str(lay)]
        b = weights["b" + str(lay)]
        z = np.dot(W, A) + b
        if lay == L:
            A = softmax(z)
        else:
            A = np.tanh(z)
            mask = np.random.rand(A.shape[0], A.shape[1])
            mask = (mask < keep_prob).astype(int)
            A *= mask
            A = A / keep_prob
            cache["D" + str(lay)] = mask
        cache["A" + str(lay)] = A
    return cache
