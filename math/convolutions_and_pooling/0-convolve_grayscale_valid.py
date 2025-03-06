#!/usr/bin/env python3
""" module containing function that performs a same convolution on grayscale
    images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ function that performs a same convolution on grayscale images

        PARAMETERS
        ==========
            images [np.ndarray]: multiple gray scale images of shape (m, h, w)
                    m - number of images
                    h - height in pixels of the images
                    w - width in pixels of the images
            kernel [np.ndarray]: kernel for the convolution of shape (kh, kw)
                    kh - height of the kernel
                    kw - width of the kernel

        RETURNS
        =======
            numpy.ndarray containing the convolved images
    """
    kern1, kern2 = kernel.shape
    m, h, w = images.shape
    op_size1 = (h - kern1) + 1
    op_size2 = (w - kern2) + 1
    op = np.zeros((m, op_size1, op_size2))
    for row in range(op_size1):
        for col in range(op_size2):
            cur = images[:, row:row+kern1, col:col+kern2]
            op[:, row, col] = np.sum(np.multiply(cur, kernel), axis=(1, 2))
    return op
