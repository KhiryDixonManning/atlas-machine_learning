#!/usr/bin/env python3
"""Module containing a function that normalizes an unactivated output of a
   neural network using batch normalization."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
       normalization.

    Parameters:
        Z (np.ndarray): Array of shape (m, n) that should be normalized,
                         where m is the number of data points and n is the
                         number of features.
        gamma (np.ndarray): Array of shape (1, n) containing the scales used
                             for batch normalization.
        beta (np.ndarray): Array of shape (1, n) containing the offsets used
                            for batch normalization.
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        np.ndarray: The output after applying batch normalization.
    """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    z_norm = (Z - mean) / np.sqrt(var + epsilon)
    out = gamma * z_norm + beta
    return out
