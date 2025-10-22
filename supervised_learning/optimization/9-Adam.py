#!/usr/bin/env python3
"""Module containing a function that updates a variable in place
   using the Adam optimization algorithm."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm.

    Parameters:
        alpha (float): The learning rate.
        beta1 (float): The weight used for the first moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (np.ndarray): The variable to be updated.
        grad (np.ndarray): The gradient of the variable.
        v (np.ndarray): The previous first moment of the variable.
        s (np.ndarray): The previous second moment of the variable.
        t (int): The time step used for bias correction.

    Returns:
        new_var (np.ndarray): The updated variable.
        new_mom1 (np.ndarray): The new first moment.
        new_mom2 (np.ndarray): The new second moment.
    """
    # Update the first moment (m)
    new_mom1 = beta1 * v + (1 - beta1) * grad

    # Update the second moment (v)
    new_mom2 = beta2 * s + (1 - beta2) * (grad ** 2)

    # Apply bias correction to the first and second moments
    m1 = new_mom1 / (1 - np.power(beta1, t))
    m2 = new_mom2 / (1 - np.power(beta2, t))

    # Update the variable using the Adam optimization rule
    new_var = var - alpha * m1 / (np.sqrt(m2) + epsilon)

    return new_var, new_mom1, new_mom2
