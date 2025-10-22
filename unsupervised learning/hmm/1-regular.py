#!/usr/bin/env python3
""" module containing function that that determines the steady state
    probabilities of a regular markov chain """
import numpy as np


def regular(p):
    """ function that that determines the steady state probabilities of a
        regular markov chain

        Parameters
        ----------
        P : numpy.ndarray
            Transition matrix of shape (n x n) where
            n is the number of states.

        Returns
        -------
        sstate_prob : numpy.ndarray or None
            the steady state probabilities, or None on failure

        """
    # check to make sure all values in matrix are positive
    if p.all() > 0:

        # get the eigen values and eigen vectors
        values, vectors = np.linalg.eig(p.T)

        # find the index of the eigenvalue that equals 1
        idx = list(np.round(values, decimals=1)).index(1)

        # get the vector associated with the value of 1
        sstate_prob = vectors[:, idx]

        # normalize so probabilities add to 1
        sstate_prob /= sum(sstate_prob)

        return np.array([sstate_prob])
    return None
