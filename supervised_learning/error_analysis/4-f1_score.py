#!/usr/bin/env python3
""" module containing function that calculates the F1 score of a confusion
    matrix """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ function that calculates the F1 score of a confusion matrix


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
                            F1 score of each class
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * ((recall * prec) / (recall + prec))
