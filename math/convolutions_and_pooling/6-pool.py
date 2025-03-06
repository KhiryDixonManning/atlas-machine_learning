#!/usr/bin/env python3
""" module containing function that """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ function that

        PARAMETERS
        ==========
            []:
            []:

        RETURNS
        =======
            []:
    """
    k_h, k_w = kernel_shape
    m, h, w, c = images.shape
    sh, sw = stride
    op_h = int((h - k_h) / sh) + 1
    op_w = int((w - k_w) / sw) + 1
    pooled = np.zeros((m, op_h, op_w, c))
    for row in range(op_h):
        for col in range(op_w):
            cur = images[:, row*sh:row*sh+k_h, col*sw:col*sw+k_w]
            if mode == 'avg':
                pooled[:, row, col] = np.mean(cur, axis=(1, 2))
            else:
                pooled[:, row, col] = np.max(cur, axis=(1, 2))
    return pooled
