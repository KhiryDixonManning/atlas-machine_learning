#!/usr/bin/env python3
""" module containing two functions that save and load a kera's model
    configuration in JSON format """
import tensorflow.keras as K


def save_config(network, filename):
    """ function that saves a model's configuration in JSON format

        PARAMETERS
        ==========
            network [keras model]: the model whose configuration should
                                    be saved
            filename [str]: the path of the file that the configuration
                            should be saved to

        RETURNS
        =======
            None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """ function that loads a model with a specific configuration

        PARAMETERS
        ==========
            filename [str]: the path of the file containing the models
                            configuration in JSON format

        RETURNS
        =======
            the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
