#!/usr/bin/env python3
""" module containing function that trains a model using mini-batch
    gradient descent and analyzes validation data """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ function that trains a model using mini-batch gradient descent
        and analyzes validation data

        PARAMETERS
        ==========
            network [keras model]: the model to train
            data [np.ndarray]: the input data (m, nx)
            labels [np.ndarray]: the labels of data (m, classes)
            batch_size [int]: the size of the batch used for mini-batch
                                gradient descent
            epochs [int]: the number of passes through data for mini-batch
                            gradient descent
            verbose [boolean]: determines if output should be printed during
                                training
            shuffle [boolean]: determines whether to shuffle the batches
                                every epoch.
            validation_data [np.ndarray]: data to validate the model with

        RETURNS
        =======
            History object
    """
    return network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
