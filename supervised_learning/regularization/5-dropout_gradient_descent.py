#!/usr/bin/env python3
""" module containing function that updates the weights of a neural network
    with Dropout regularization using gradient descent """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ function that updates the weights of a neural network with Dropout
        regularization using gradient descent


        PARAMETERS
        ==========
        Y [np.ndarray]: one-hot array of shape (classes, m) that contains the
                        correct labels for the data
                        > classes [int]: number of classes
                        > m [int]: number of data points

        weights [dict]: the weights and biases of the neural network

        cache [dict]: outputs of each layer of the neural network

        alpha [float]: learning rate

        L [int]: number of layers in the network

        keep_prob [?]: probability that a node will be kept


        RETURNS
        =======
        None
    """
    N = Y.shape[1]
    adj = weights
    A_cur = cache["A" + str(L)]
    dz = (A_cur - Y)

    for lay in range(L, 0, -1):
        W_cur = adj["W" + str(lay)]
        A_cur = cache["A" + str(lay)]
        A_prev = cache["A" + str(lay - 1)]
        b_cur = adj["b" + str(lay)]

        dW1 = ((1 / N) * np.dot(dz, A_prev.T))
        db1 = (1 / N) * np.sum(dz, axis=1, keepdims=True)

        dg = ((1 - A_prev ** 2))
        dz = (np.dot(W_cur.T, dz)) * dg

        if lay > 1:
            dz = dz * cache["D" + str(lay - 1)]
            dz = dz / keep_prob

        adj["W" + str(lay)] = W_cur - (alpha * dW1)
        adj["b" + str(lay)] = b_cur - (alpha * db1)

    weights = adj
