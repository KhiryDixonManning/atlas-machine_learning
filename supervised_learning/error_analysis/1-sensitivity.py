#!/usr/bin/env python3
""" module containing function that calculates the sensitivity for each class
    in a confusion matrix """
import numpy as np


def sensitivity(confusion):
    """ function that calculates the sensitivity for each class in a
        confusion matrix


        PARAMETERS
        ==========
        confusion [np.ndarray]: confusion array of shape (classes, classes)
                                with row indices representing the correct
                                labels and column indices representing the
                                predicted labels
                                    > classes [int]: number of classes


        RETURNS
        =======
        [numpy.ndarray]: array of shape (classes,) containing the
                            sensitivity of each class
    """
    sense = np.zeros(len(confusion))
    for i in range(len(confusion)):
        sense[i] += round(confusion[i][i] /
                          np.sum(confusion[i]), 8)
    return sense
