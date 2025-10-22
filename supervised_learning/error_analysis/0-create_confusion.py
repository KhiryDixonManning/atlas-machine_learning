#!/usr/bin/env python3
""" module containing function that creates a confusion matrix and a function
    that decodes a one-hot array """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ function that creates a confusion matrix


        PARAMETERS
        ==========
        labels [np.ndarray]: one-hot matrix of shape (m, class) containing
                                the correct labels for each data point
                                    > m [int]:  number of data points
                                    > classes [int]: number of classes

        logits [np.ndarray]: one-hot matrix of shape (m, class) containing
                                the logitsicted labels
                                    > m [int]:  number of data points
                                    > classes [int]: number of classes


        RETURNS
        =======
        [np.ndarray]: confusion array of shape (classes, classes) with row
                        indices representing the correct labels and column
                        indices representing the logitsicted labels
    """
    class_num = labels.shape[1]
    class_list = np.arange(0, class_num, 1)
    labels = decode(labels)
    logits = decode(logits)
    confusion = np.zeros((class_num, class_num))
    for i in range(class_num):
        for j in range(class_num):
            confusion[i, j] = np.sum((labels == class_list[i]) &
                                     (logits == class_list[j]))

    return confusion


def decode(one_hot):
    """ function that decodes a one-hot array """
    decoded = []
    for i in range(one_hot.shape[0]):
        for j in range(one_hot.shape[1]):
            if one_hot[i][j] == 1:
                decoded.append(j)

    return decoded
