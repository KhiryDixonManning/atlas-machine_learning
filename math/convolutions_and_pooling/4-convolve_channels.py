#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ function that

        PARAMETERS
        ==========
            []:
            []:

        RETURNS
        =======
            []:
    """
    kern1, kern2, kern3 = kernel.shape
    m, h, w, c = images.shape
    sh, sw = stride
    if padding == 'same':
        op = np.zeros((images.shape[0], h, w))
        p_h = int(((h * sh) - h + kern1 - sh) / 2) + 1
        p_w = int(((w * sw) - w + kern2 - sw) / 2) + 1
        pad = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                     mode='constant')
        for row in range(h):
            for col in range(w):
                cur = pad[:, row*sh:row*sh+kern1, col*sw:col*sw+kern2, :]
                op[:, row, col] = np.sum(np.multiply(cur, kernel),
                                         axis=(1, 2, 3))
        return op

    elif padding == 'valid':
        op_size1 = int(np.ceil((h - kern1 + 1) / sh))
        op_size2 = int(np.ceil((w - kern2 + 1) / sw))
        op = np.zeros((images.shape[0], op_size1, op_size2))
        for row in range(op_size1):
            for col in range(op_size2):
                cur = images[:, row*sh:row*sh+kern1, col*sw:col*sw+kern2, :]
                op[:, row, col] = np.sum(np.multiply(cur, kernel),
                                         axis=(1, 2, 3))
        return op

    elif isinstance(padding, tuple):
        ph, pw = padding
        new_h = int(round((h - kern1 + (2 * ph) + sh) / sh)) - 1
        new_w = int(round((w - kern2 + (2 * pw) + sw) / sw))
        op = np.zeros((m, new_h, new_w))
        pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')
        for row in range(new_h):
            for col in range(new_w):
                cur = pad[:, row*sh:row*sh+kern1, col*sw:col*sw+kern2, :]
                op[:, row, col] = np.sum(np.multiply(cur, kernel),
                                         axis=(1, 2, 3))
        return op
