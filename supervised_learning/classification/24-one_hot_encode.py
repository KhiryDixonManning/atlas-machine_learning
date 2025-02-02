#!/usr/bin/env python3
""" module containing function that decodes a one hot encoded vector """
import numpy as np


def one_hot_encode(Y, classes):
    """ function that decodes a one hot encoded vector """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes < 2 or classes < Y[max(Y - 1)]:
        return None
    encoded = np.zeros((classes, len(Y)))
    for x, y in enumerate(Y):
        encoded[y][x] = 1
    return encoded
