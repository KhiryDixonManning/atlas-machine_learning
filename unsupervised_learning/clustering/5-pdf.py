#!/usr/bin/env python3\
""" module containing function that calculates the probability density
function of a Gaussian distribution. """
import numpy as np


def pdf(X, m, S):
    """ function that calculates the probability density function of a
        Gaussian distribution.

        Parameters
        ----------
        X : numpy.ndarray
            Data points with shape (n, d).
        m : numpy.ndarray
            Mean of the distribution with shape (d,).
        S : numpy.ndarray
            Covariance of the distribution with shape (d, d).

        Returns
        -------
        P : numpy.ndarray or None
            PDF values for each data point with shape (n,).
            Returns None on failure.
    """
    # check validity of X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2:
        return None

    # check validity of m
    if not isinstance(m, type(np.array([]))) or len(
            m.shape) > 1 or m.shape[0] != X.shape[1]:
        return None

    # check validity of S
    if not isinstance(S, type(np.array([]))) or len(
            S.shape) < 2 or isinstance(S[0][0], type(np.array([]))) \
            or S.shape[0] != S.shape[1]:
        return None

    # get shapes
    k = len(m)

    # reshape mean to make suitable in computation
    mm = m.reshape(k, 1)

    # find constant
    const = (2 * np.pi) ** k

    # determinate and inverse of covariance mat
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    # scaling
    scale = 1.0 / np.sqrt(const * det)

    # mean shifted X
    diff = (X - mm.T)

    # calc exponential term
    exp_term = -0.5 * (np.dot(diff, inv).dot(diff.T))

    # create mask to extract diagonal values
    row, col = exp_term.shape
    diag = np.eye(row, col, dtype=bool)
    exp_diag = exp_term[diag]

    # combine
    res = np.log(scale) + exp_diag
    p_d_f = np.exp(res)

    # set min value to 1e-300
    p_d_f[p_d_f < 1e-300] = 1e-300

    return p_d_f
