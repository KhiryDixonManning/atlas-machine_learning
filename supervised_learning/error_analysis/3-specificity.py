#!/usr/bin/env python3
""" module containing function that calculates the specificity for each class
    in a confusion matrix """
import numpy as np


def specificity(confusion):
    """ function that calculates the specificity for each class in a
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
                            specificity of each class
    """
    specific = np.zeros(len(confusion))
    for x in range(len(confusion)):
        tnfp = 0
        tn = 0
        for i in range(len(confusion)):
            for j in range(len(confusion)):
                if i != x:
                    tnfp += confusion[i][j]
                    if j != x:
                        tn += confusion[i][j]
        specific[x] += round((tn / (tnfp)), 8)
    return specific
