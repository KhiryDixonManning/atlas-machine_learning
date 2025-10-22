#!/usr/bin/env python3
""" module containing function that trains a model using mini-batch gradient
    descent, learning rate decay, analyzes validation data, performs early
    stopping """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """ Function that trains a model using mini-batch gradient descent,
        learning rate decay, analyzes validation data, performs earlystopping

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
        shuffle [boolean]: Determines whether to shuffle the batches every
                            epoch.
        validation_data [np.ndarray]: Data to validate the model with.
        early_stopping [boolean]: Indicates whether early stopping should
                                    be used.
        patience [int]: The patience used for early stopping.
        learning_rate_decay [boolean]: Indicates whether learning rate decay
                                        should be used.
        alpha [float]: The initial learning rate.
        decay_rate [float]: The decay rate.

    RETURNS
    =======
        History object
    """
    cbs = []

    def scheduler(epoch):
        return alpha / (1 + decay_rate * epoch)

    if validation_data:
        if early_stopping:
            es = K.callbacks.EarlyStopping(patience=patience)
            cbs.append(es)
        if learning_rate_decay:
            lr = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
            cbs.append(lr)
        return network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                           verbose=verbose, shuffle=shuffle,
                           callbacks=cbs,
                           validation_data=validation_data)
    return network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle)
