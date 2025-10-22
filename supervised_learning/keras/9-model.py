#!/usr/bin/env python3
""" module containing two functions that saves and loads an entire model """
import tensorflow.keras as K


def save_model(network, filename):
    """ function that saves an entire model.

    Parameters:
        network [keras model]: The model to save.
        filename [str]: path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """ function that loads an entire model.

    Parameters:
        filename [str]: path of the file that the model should be loaded from.

    Returns:
        The loaded model
    """
    return K.models.load_model(filename)
