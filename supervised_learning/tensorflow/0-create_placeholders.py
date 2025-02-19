#!/usr/bin/env python3
"""Module containing a function that returns two placeholders, x and y,
   for a neural network."""

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Function that constructs TensorFlow placeholders for X and Y.

    Parameters:
        nx (int): Number of feature columns in our data.
        classes (int): Number of classes in our classifier.

    Returns:
        tuple: A tuple containing:
            - x: placeholder for input data.
            - y: placeholder for one-hot labels for input data.
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name="y")
    return x, y
