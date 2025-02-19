#!/usr/bin/env python3
""" module containing function that creates the training operation for
    the network"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ function that creates the training operation for the network

        Parameters:
            loss [float] -  loss of the networks prediction
            alpha [float] - learning rate

        Returns:
            [operation] - an operation that trains the network using gradient
                            descent
        """
    optim = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train = optim.minimize(loss)
    return train
