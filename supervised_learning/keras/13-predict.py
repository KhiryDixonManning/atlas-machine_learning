#!/usr/bin/env python3
""" module containing function that makes a prediction using a neural
    network """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ function that makes a prediction using a neural network

        PARAMETERS
        ==========
            network [keras model]: network model to make the prediction with
            data [np.ndarray]: the input data to make the prediction with
            verbose [boolean]: the path of the file containing the model's
                                configuration in JSON format

        RETURNS
        =======
            prediction for the data
    """
    return network.predict(data, verbose=verbose)
