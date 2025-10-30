#!/usr/bin/env python3
""" module containing function that determines if a markov chain is
    absorbing """
import numpy as np


def absorbing(P):
    """ function that determines if a markov chain is absorbing

        Parameters
        ----------
        P : numpy.ndarray
            Transition matrix of shape (n x n) where
            n is the number of states.
            P[i, j] is the probability of transitioning from
            state i to state j

        Returns
        -------
        bool
            True if markov chain is absorbing, otherwise False

        """
    # get size of P
    n = len(P)

    # if there are any 1's in the diagonal, continue
    if list(np.diag(P)).count(1.0) > 0:

        # work with only the first element as im using recursion
        if P[0][0] == 1:

            # if size of matrix is 2, chain is absorbing
            if n == 2:
                return True

            # check if any value in the same column as the 1 is above 0,
            # meaning it is possible to move to that state from another state
            if P[1:, 0].any() > 0:

                # if it is possible to move to that state from another state,
                # check to make sure that the other state is also reachable
                # from any other state besides the absorbing state
                if P[2:, 1].any() > 0:
                    return True

                # if not, the chain is not absorbing because the absorbing
                # state is inaccessable
                else:
                    return False

            # if it is not possible to get to that state from any other state,
            # recursively call this function to check the next submatrix
            else:
                if absorbing(P[1:, 1:]):
                    return True

    # return false on failure or if there are no 1's in diagonal
    return False
