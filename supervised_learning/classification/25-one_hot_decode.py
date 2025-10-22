#!/usr/bin/env python3
""" module containing function that decodes a one hot encoded vector """
import numpy as np


def one_hot_decode(one_hot):
    """ function that decodes a one hot encoded vector """
    decoded = []
    try:
        for x in range(one_hot.shape[1]):
            for y in range(one_hot.shape[0]):
                if one_hot[y][x] == 1:
                    decoded.append(y)
        return np.array(decoded, int)
    except Exception as e:
        return None
