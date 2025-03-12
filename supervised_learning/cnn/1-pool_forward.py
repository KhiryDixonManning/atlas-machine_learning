#!/usr/bin/env python3
""" module containing function that performs forward propagation over a
    pooling layer of a neural network."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Function that performs forward propagation over a pooling layer
        of a neural network.

    PARAMETERS
    ==========
        A_prev [numpy.ndarray]: Output of the previous layer.
                                Shape (m, h_prev, w_prev, c_prev)
            m - number of examples
            h_prev - height of the previous layer
            w_prev - width of the previous layer
            c_prev - number of channels in the previous layer

        kernel_shape [tuple]: Size of the kernel for the pooling.
                                Tuple of (kh, kw)
            kh - kernel height
            kw - kernel width

        stride [tuple]: Strides for the pooling. Tuple of (sh, sw)
            sh - stride for the height
            sw - stride for the width

        mode [str]:
            Indicates whether to perform maximum or average pooling.
            Can be 'max' for maximum pooling or 'avg' for average pooling.

    RETURNS
    =======
        Output of the pooling layer.
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    op_h = int((h - kh) / sh) + 1
    op_w = int((w - kw) / sw) + 1
    pooled = np.zeros((m, op_h, op_w, c))
    for row in range(op_h):
        for col in range(op_w):
            cur = A_prev[:, row*sh:row*sh+kh, col*sw:col*sw+kw]
            if mode == 'avg':
                pooled[:, row, col] = np.mean(cur, axis=(1, 2))
            else:
                pooled[:, row, col] = np.max(cur, axis=(1, 2))
    return pooled
