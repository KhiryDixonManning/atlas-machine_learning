#!/usr/bin/env python3
"""Module containing a function that creates a learning rate decay operation
   in TensorFlow using inverse time decay."""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in TensorFlow using inverse
       time decay.

    Parameters:
        alpha (float): The initial learning rate.
        decay_rate (float): The weight used to determine the rate at which
                            alpha will decay.
        global_step (int): The number of passes of gradient descent that
                           have elapsed.
        decay_step (int): The number of passes of gradient descent that
                          should occur before alpha is decayed further.

    Returns:
        tf.Operation: The learning rate decay operation.
    """
    optim = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return optim
