#!/usr/bin/env python3
""" module containing function that calculates the cost of a neural network
    with L2 regularization """
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ function that calculates the cost of a neural network with L2
        regularization


        PARAMETERS
        ==========
        cost [tensor]: the cost of the network without L2 regularization


        RETURNS
        =======
        [tensor]: cost of the network accounting for L2 regularization
    """
    reg_loss = tf.compat.v1.losses.get_regularization_losses()
    return cost + reg_loss
