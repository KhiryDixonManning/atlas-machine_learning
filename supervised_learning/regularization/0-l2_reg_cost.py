#!/usr/bin/env python3
""" module containing function that calculates the cost of a neural network
    with L2 regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ function that calculates the cost of a neural network with L2
        regularization


        PARAMETERS
        ==========
        cost [?]: cost of the network without L2 regularization

        lambtha [?]: regularization parameter

        weights [dict]: weights and biases (numpy.ndarrays) of the neural
                        network

        L [?]: number of layers in the neural network

        m [int]: number of data points used


        RETURNS
        =======
        [?]: cost of the network accounting for L2 regularization
    """
    w = 0
    for i in range(1, L + 1):
        w += np.sum(np.square(weights['W' + str(i)]))
    return cost + (lambtha / (2 * m)) * w
