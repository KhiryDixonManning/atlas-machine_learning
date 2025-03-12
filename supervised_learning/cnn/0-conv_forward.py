#!/usr/bin/env python3
""" module containing Function that performs a same convolution on
    grayscale images. """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Function that performs a same convolution on grayscale images.

    PARAMETERS
    ==========
        A_prev [numpy.ndarray]: Multiple gray scale images of shape (m, h, w)
            m - number of images
            h - height in pixels of the images
            w - width in pixels of the images

        W [numpy.ndarray]: Kernel for the convolution of shape (kh, kw)
            kh - height of the kernel
            kw - width of the kernel

        b [numpy.ndarray]: Biases applied to the convolution.

        activation [func]: Activation function applied to the convolution.

        padding [str]: Type of padding used. Either 'same' or 'valid'.

        stride [tuple]: Strides for the convolution (sh, sw)
                sh - stride for the height
                sw - stride for the width

    RETURNS
    =======
        output of convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h = int(((h_prev * sh) - h_prev + kh - sh) / 2)
        pad_w = int(((w_prev * sw) - w_prev + kh - sw) / 2)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                 (0, 0)), mode='constant')
    out_h = int(1 + (h_prev + (2 * pad_h) - kh) / sh)
    out_w = int(1 + (w_prev + (2 * pad_w) - kw) / sw)
    op = np.zeros((m, out_h, out_w, c_new))
    for row in range(out_h):
        for col in range(out_w):
            for i in range(c_new):
                cur = A_prev[:, row*sh:row*sh+kh, col*sw:col*sw+kw]
                op[:, row, col, i] = (np.sum(
                    np.multiply(cur, W[:, :, :, i]),
                    axis=(1, 2, 3))) + b[:, :, :, i]
    return activation(op)
