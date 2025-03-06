#!/usr/bin/env python3
""" module containing function that updates the weights and biases of a
    neural network using gradient descent with L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ function that updates the weights and biases of a neural network
        using gradient descent with L2 regularization


        PARAMETERS
        ==========
        Y [np.ndarray]: one-hot array of shape (classes, m) that contains the
                        correct labels for the data
                        > classes [int]: number of classes
                        > m [int]: number of data points

        weights [dict]: the weights and biases of the neural network

        cache [dict]: outputs of each layer of the neural network

        alpha [float]: learning rate

        lambtha [?]:  L2 regularization parameter

        L [int]: number of layers of the network


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

        dW1 = ((1 / N) * np.dot(dz, A_prev.T)) + (W_cur * (lambtha / N))
        db1 = (1 / N) * np.sum(dz, axis=1, keepdims=True)

        dg = ((1 - A_prev ** 2))
        dz = (np.dot(W_cur.T, dz)) * dg

        adj["W" + str(lay)] = W_cur - (alpha * dW1)
        adj["b" + str(lay)] = b_cur - (alpha * db1)

    weights = adj
