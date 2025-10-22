#!/usr/bin/env python3
""" module containing two functions that saves and loads a models weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ function that saves a models weights.

    Parameters:
        network [keras model]: The model whose weights should be saved.
        filename [str]: Path of the file that the weights should be saved to.
        save_format [str]: The format in which the weights should be saved.
    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ function that loads a models weights.

    Parameters:
        network [keras model]: The model to which the weights should be loaded.
        filename [str]: path of file that the weights should be loaded from

    Returns:
        None
    """
    network.load_weights(filename)
