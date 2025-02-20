#!/usr/bin/env python3
"""Module containing a function that updates the learning rate using
   inverse time decay in NumPy."""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in NumPy.

    Parameters:
        alpha (float): The initial learning rate.
        decay_rate (float): The weight used to determine the rate at which
                            alpha will decay.
        global_step (int): The number of passes of gradient descent that
                           have elapsed.
        decay_step (int): The number of passes of gradient descent that
                          should occur before alpha is decayed further.

    Returns:
        float: The updated learning rate.
    """
    if global_step < decay_step:
        return alpha
    else:
        step = int(global_step / decay_step)
        alpha = alpha / (1 + decay_rate * step)
    return alpha
