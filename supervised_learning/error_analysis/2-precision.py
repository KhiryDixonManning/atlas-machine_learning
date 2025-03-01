#!/usr/bin/env python3
""" module containing function that calculates the precision for each class
    in a confusion matrix """
import numpy as np


def precision(confusion):
    """ function that calculates the precision for each class in a confusion
        matrix


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
                            precision of each class
    """
    precision = np.zeros(len(confusion))
    for i in range(len(confusion)):
        fp = 0
        for j in range(len(confusion)):
            fp += confusion[j][i]
        precision[i] += round(confusion[i][i] / np.sum(fp), 8)
    return precision
