#!/usr/bin/env python3
""" module containing function that finds the best number of clusters for a
    Gaussian Mixture Model (GMM) using the Bayesian Information Criterion
    (BIC). """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ function that finds the best number of clusters for a Gaussian
        Mixture Model (GMM) using the Bayesian Information Criterion (BIC).

        Parameters:
        -----------
        X : numpy.ndarray
            Input data set of shape (n, d).
        kmin : int, optional
            Minimum number of clusters to check for (inclusive). Default is 1.
        kmax : int, optional
            Maximum number of clusters to check for (inclusive). If None,
            set to the maximum number of clusters possible. Default is None.
        iterations : int, optional
            Maximum number of iterations for the EM algorithm. Default is 1000.
        tol : float, optional
            Tolerance for the EM algorithm. Default is 1e-5.
        verbose : bool, optional
            Determines if the EM algorithm should print information to
            the standard output. Default is False.

        Returns:
        --------
        Tuple
            (best_k, best_result, l, b) or (None, None, None, None) on failure.
        best_k : int
            The best value for k based on its BIC.
        best_result : tuple
            Tuple containing pi, m, S.
            pi : numpy.ndarray
                Cluster priors for the best number of clusters of shape (k,).
            m : numpy.ndarray
                Centroid means for the best number of clusters of shape (k, d).
            S : numpy.ndarray
                Covariance matrices for the best number of clusters of shape
                (k, d, d).
        l_l : numpy.ndarray
            Log likelihood for each cluster size tested of
            shape (kmax - kmin + 1).
        b : numpy.ndarray
            BIC value for each cluster size tested of
            shape (kmax - kmin + 1).
    """
    # check validity of X
    if not isinstance(X, type(np.array([]))) or len(
            X.shape) < 2 or isinstance(X[0][0], type(np.array([]))):
        return None, None, None, None

    # check validity of kmin and kmax
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmax is not None:
        if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
            return None, None, None, None

    # check validity of iterations
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    # check validity of tol and verbose
    if not isinstance(tol, float) or tol <= 0 or \
            not isinstance(verbose, bool):
        return None, None, None, None

    bic_s = []
    l_s = []
    for i in range(kmin, kmax):

        # calc EM fur each cluster sie
        pi, m, S, g, l_l = expectation_maximization(
            X, i, iterations, tol, verbose)

        # get shapes
        k = pi.shape[0]
        _, d = m.shape
        n, _ = X.shape

        # calc number of params
        p = (k * d) + (k * d * d) + (k - 2)

        # BIC calc
        bic = p * np.log(n) - 2 * l_l

        # store best results
        if i == 1:
            best_log = l_l
            best_res = (pi, m, S)

        if l_l <= best_log:
            best_log = l_l
            best_k = i
            best_res = (pi, m, S)

        # add BIC and likelihood to list
        bic_s.append(round(bic, 8))
        l_s.append(np.round(l_l, 8))

    return best_k, best_res, np.array(l_s), np.array(bic_s)
