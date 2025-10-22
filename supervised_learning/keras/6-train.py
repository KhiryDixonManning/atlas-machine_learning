#!/usr/bin/env python3
""" module containing function that trains a model using mini-batch gradient
    descent, analyzes validation data, and performs earlystopping """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ Function that trains a model using mini-batch gradient descent,
    analyzes validation data, and performs earlystopping

    PARAMETERS
    ==========
        network [keras model]: The model to train.
        data [np.ndarray]: The input data (m, nx).
        labels [np.ndarray]: The labels of data (m, classes).
        batch_size [int]: The size of the batch used for mini-batch gradient
                            descent.
        epochs [int]: The number of passes through data for mini-batch
                        gradient descent.
        verbose [boolean]: Determines if output should be printed during
                            training.
        shuffle [boolean]: Determines whether to shuffle batches every epoch.
        validation_data [np.ndarray]: Data to validate the model with.
        early_stopping [boolean]: Indicates whether early stopping should
                                    be used.
        patience [int]: The patience used for early stopping.

    RETURNS
    =======
        History object
    """
    if validation_data and early_stopping:
        callback = K.callbacks.EarlyStopping(patience=patience)
        return network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                           verbose=verbose, shuffle=shuffle,
                           callbacks=[callback],
                           validation_data=validation_data)
    return network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
