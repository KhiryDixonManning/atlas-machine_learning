#!/usr/bin/env python3
""" module containing function that performs the expectation maximization
    for a Gaussian Mixture Model (GMM). """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ function that performs the expectation maximization for a Gaussian
        Mixture Model (GMM).

        Parameters:
        -----------
        X : numpy.ndarray
            Data set with shape (n, d).
        k : int
            Number of clusters.
        iterations : int, optional
            Maximum number of iterations for the algorithm. Default is 1000.
        tol : float, optional
            Tolerance of the log likelihood, used to determine early stopping.
            If the difference is less than or equal to tol, the algorithm
            stops. Default is 1e-5.
        verbose : bool, optional
            Determines if information about the algorithm should be printed.
            If True, prints Log Likelihood after {i} iterations: {l} every 10
            iterations and after the last iteration. Default is False.

        Returns:
        --------
        Tuple
            (pi, m, S, g, l) or (None, None, None, None, None) on failure.
        pi : numpy.ndarray
            Priors for each cluster of shape (k,).
        m : numpy.ndarray
            Centroid means for each cluster of shape (k, d).
        S : numpy.ndarray
            Covariance matrices for each cluster of shape (k, d, d).
        g : numpy.ndarray
            Probabilities for each data point in each cluster of shape (k, n).
        l : float
            Log likelihood of the model.
    """
    # check validity of X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None, None

    # check validity of k
    if type(k) is not int or k <= 0:
        return None

    # check validity of iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # check validity of tol and verbose
    if not isinstance(tol, float) or tol <= 0 or \
            not isinstance(verbose, bool):
        return None, None

    message = 'Log Likelihood after {} iterations: {}'
    pi, m, S = initialize(X, k)
    for i in range(iterations):
        if i > 0:
            prev_likelihood = likelihood
        g, likelihood = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose is True:
            print(message.format(i, round(likelihood, 5)))
        if i > 0 and abs(likelihood - prev_likelihood) <= tol:
            break
        pi, m, S = maximization(X, g)
    if verbose is True:
        print(message.format(i, round(likelihood, 5)))
    return pi, m, S, g, likelihood
