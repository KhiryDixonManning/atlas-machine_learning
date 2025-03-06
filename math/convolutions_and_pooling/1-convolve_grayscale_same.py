#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ function that

        PARAMETERS
        ==========
            []:
            []:

        RETURNS
        =======
            []:
    """
    kern1 = kernel.shape[0]
    kern2 = kernel.shape[1]
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    pad_tb = int(round((kern1 - 1) / 2))
    pad_lr = int(round((kern2 - 1) / 2))
    op = np.zeros((m, h, w))
    pad = np.pad(images, ((0, 0), (pad_tb, pad_tb), (pad_lr, pad_lr)),
                 mode='constant')
    for row in range(h):
        for col in range(w):
            cur = pad[:, row:row+kern1, col:col+kern2]
            op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
    return op
