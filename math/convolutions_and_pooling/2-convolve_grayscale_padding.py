#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    ph = padding[0]
    pw = padding[1]
    new_h = (h - kern1 + (2 * ph) + 1)
    new_w = (w - kern2 + (2 * pw) + 1)
    op = np.zeros((m, new_h, new_w))
    pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                 mode='constant')
    for row in range(new_h):
        for col in range(new_w):
            cur = pad[:, row:row+kern1, col:col+kern2]
            op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
    return op
