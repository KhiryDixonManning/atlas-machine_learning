#!/usr/bin/env python3
""" module containing function that tests a neural network """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ function that tests a neural network

        PARAMETERS
        ==========
            network [keras model]: network model to test
            data [np.ndarray]: input data to test the model with
            labels [np.ndarray]: the correct one-hot labels of data
            verbose [boolean]: determines if output should be printed during
            the testing process

        RETURNS
        =======
            the loss and accuracy of the model with the testing data,
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
