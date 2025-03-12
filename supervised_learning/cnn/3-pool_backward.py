#!/usr/bin/env python3
""" module containing Function that performs back propagation over a
    pooling layer of a neural network. """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Function that performs back propagation over a pooling layer of a
        neural network.

    PARAMETERS
    ==========
        dA [numpy.ndarray]: Partial derivatives with respect to the output of
                            pooling layer of shape (m, h_new, w_new, c_new)
            m - number of examples
            h_new - height of the output
            w_new - width of the output
            c_new - number of channels

        A_prev [numpy.ndarray]: Output of the previous layer of shape
                                (m, h_prev, w_prev, c)
            m - number of examples
            h_prev - height of the previous layer
            w_prev - width of the previous layer
            c - number of channels

        kernel_shape [tuple]: Size of the kernel for the pooling, (kh, kw)
            kh - kernel height
            kw - kernel width

        stride [tuple]: Strides for the pooling, (sh, sw)
            sh - stride for the height
            sw - stride for the width

        mode [str]: Indicates whether to perform maximum or average pooling.
            Can be 'max' for maximum pooling or 'avg' for average pooling.

    RETURNS
    =======
        Gradient of the cost with respect to the activation of the previous
        layer.
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out = np.zeros(A_prev.shape)
    for sample in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for i in range(c_new):
                    if mode == 'max':
                        cur = A_prev[sample, row*sh:row*sh+kh,
                                     col*sw:col*sw+kw, i]
                        maxed = (cur == np.max(cur))
                        out[sample, row*sh:row*sh+kh, col*sw:col*sw+kw,
                            i] += dA[sample, row, col, i] * maxed
                    else:
                        da = dA[sample, row, col, i]
                        avg = da / (kh * kw)
                        out[sample, row*sh:row*sh+kh, col*sw:col*sw+kw,
                            i] += np.ones(kernel_shape) * avg
    return out
