#!/usr/bin/env python3
""" module containing function that returns two placeholders x and y
    for a neural network """
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ function that contructs TensorFlow placeholders for X and Y

        Parameters:
            nx [int] - number of feature columns in our data
            classes [int] - number of classes in our classifier

        Returns:
            [Tensor] placeholders for x and y
                x - placeholder for input data
                y - placeholder for one-hot labels for input data
        """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")
    return x, y
