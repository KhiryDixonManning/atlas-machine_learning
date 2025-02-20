#!/usr/bin/env python3
""" module containing function that updates a variable using the
    RMSProp optimization algorithm """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the RMSProp optimization algorithm

        PARAMETERS:
            alpha [float]: the learning rate
            beta2 [float]: the RMSProp weight
            epsilon [float]: small number to avoid division by zero
            var [np.ndarray]: the variable to be updated
            grad [np.ndarray]: the gradient of var
            s [np.ndarray]: the previous second moment of var

        RETURNS:
            new_var, new_mom [np.ndarray]: updated variable and the new
                                            momentum

    """
    new_mom = beta2 * s + (1 - beta2) * (grad ** 2)
    new_var = var - (alpha * grad / (np.sqrt(new_mom) + epsilon))
    return new_var, new_mom
