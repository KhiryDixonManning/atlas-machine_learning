#!/usr/bin/env python3
""" module containing Function that performs back propagation over a
    convolutional layer of a neural network."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Function that performs back propagation over a convolutional layer of a
    neural network.

    PARAMETERS
    ==========
        dZ [numpy.ndarray]: Partial derivatives with respect to the
                            unactivated output of the convolutional layer.
            Shape (m, h_new, w_new, c_new)
            m - number of examples
            h_new - height of the output
            w_new - width of the output
            c_new - number of channels in the output

        A_prev [numpy.ndarray]: Output of the previous layer,
                                shape (m, h_prev, w_prev, c_prev)
            m - number of examples
            h_prev - height of the previous layer
            w_prev - width of the previous layer
            c_prev - number of channels in the previous layer

        W [numpy.ndarray]: Kernels for the convolution,
                            Shape (kh, kw, c_prev, c_new)
            kh - filter height
            kw - filter width

        b [numpy.ndarray]: Biases applied to the convolution.
                            Shape (1, 1, 1, c_new)

        padding [str]: Type of padding used. Either 'same' or 'valid'.

        stride [tuple]: Strides for the convolution. (sh, sw)
            sh - stride for the height
            sw - stride for the width

    RETURNS
    =======
        A tuple containing:
            - dA_prev: Gradient of the cost with respect to the activation
                        of the previous layer.
            - dW: Gradient of the cost with respect to the weights W.
            - db: Gradient of the cost with respect to the biases b.
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w),
                                 (0, 0)), mode='constant')
    dW = np.zeros(W.shape)
    dX = np.zeros(A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for sample in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for i in range(c_new):
                    dX[sample, row*sh:row*sh+kh, col*sw:col*sw+kw,
                       :] += dZ[sample, row, col, i] * W[:, :, :, i]

                    dW[:, :, :, i] += A_prev[sample,
                                             row*sh:row*sh+kh,
                                             col*sw:col*sw+kw,
                                             :] * dZ[sample, row, col, i]
    if padding == 'same':
        dX = dX[:, pad_h:dX.shape[1]-pad_h, pad_w:dX.shape[2]-pad_w, :]

    return dX, dW, db
