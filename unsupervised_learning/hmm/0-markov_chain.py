#!/usr/bin/env python3
""" function that calculates the probability of a markov chain being in a
        particular state after a specified number of iterations """
import numpy as np


def markov_chain(P, s, t=1):
    """ function that calculates the probability of a markov chain being in a
        particular state after a specified number of iterations

        Parameters
        ----------
        P : numpy.ndarray
            Transition matrix of shape (n x n) where
            n is the number of states.
        s : numpy.ndarray
            Probability of starting in each state, shape
            (1, n)
        t : int
            number of iterations the markov chain went through

        Returns
        -------
        S : numpy.ndarray or None
            probability matrix of being in a specific state after t iterations
            None on failure

        """
    # transition matrix t times
    p_t = np.linalg.matrix_power(P, t)

    # starting prob matrix times transition^t
    S = np.matmul(s, p_t)

    return S
