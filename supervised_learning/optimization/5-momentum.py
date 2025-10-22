#!/usr/bin/env python3
"""Module containing a function that updates a variable using the gradient
   descent with momentum optimization algorithm."""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with momentum
       optimization algorithm.

    Parameters:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (np.ndarray): The variable to be updated.
        grad (np.ndarray): The gradient of the variable.
        v (np.ndarray): The previous first moment of the variable.

    Returns:
        np.ndarray: The updated variable and the new momentum.
    """
    # Update the momentum term
    new_mom = beta1 * v + (1 - beta1) * grad

    # Update the variable using the momentum
    new_var = var - alpha * new_mom

    return new_var, new_mom
