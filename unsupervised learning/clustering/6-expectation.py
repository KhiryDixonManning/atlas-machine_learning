#!/usr/bin/env python3
""" function that calculates the expectation step in the EM algorithm
    for a Gaussian Mixture Model. """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ function that calculates the expectation step in the EM algorithm
        for a Gaussian Mixture Model.

        Parameters
        ----------
        X : numpy.ndarray
            Data set with shape (n, d).
        pi : numpy.ndarray
            Priors for each cluster with shape (k,).
        m : numpy.ndarray
            Centroid means for each cluster with shape (k, d).
        S : numpy.ndarray
            Covariance matrices for each cluster with shape (k, d, d).

        Returns
        -------
        g : numpy.ndarray or None
            Posterior probabilities for each data point in each cluster with
            shape (k, n).
            Returns None on failure.
        l : float or None
            Total log likelihood.
            Returns None on failure.
    """
    # check validity of X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None

    # check validity of m
    if not isinstance(m, type(np.array([]))) or len(m.shape) < 2 or isinstance(
            m[0][0], type(np.array([]))) or m.shape[1] != X.shape[1]:
        return None, None

    # check validity of pi
    if not isinstance(pi, type(np.array([]))) or len(
            pi.shape) < 1 or not np.isclose(np.sum(pi), 1):
        return None, None

    # check validity of S
    if not isinstance(S, type(np.array([]))) or \
        len(S.shape) < 3 or \
            S.shape[1] != S.shape[2] or \
            S.shape[0] != pi.shape[0]:
        return None, None

    # get shapes
    d, _ = X.shape
    k, _ = m.shape

    # empty arrays to fill with new values
    nums = np.zeros(shape=(k, d))
    e_step = np.zeros(shape=(k, d))

    # fur each cluster centroid
    for i in range(len(m)):

        # find pdf values fur each datapoint with that mean
        nums[i] = pdf(X, m[i], S[i])

        # calculate and store the numerator fur e step
        e_step[i] = np.dot(pi[i], nums[i])

    # calculate denominator using numerator values
    denom = np.sum(e_step, axis=0)

    # divide all values in estep by denom
    e_step /= denom

    # also calculate log likelihood
    return e_step, np.sum(np.log(denom))
