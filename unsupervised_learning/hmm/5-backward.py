#!/usr/bin/env python3
""" module containing function that performs the b0ackward algorithm for a
    Hidden Markov Model. """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ function that performs the backward algorithm for a
        Hidden Markov Model.

        Parameters
        ----------
        Observation : numpy.ndarray
            Array of shape (T,) containing the index of the observation.
        Emission : numpy.ndarray
            Array of shape (N, M) containing emission probabilities.
            Emission[i, j] is the probability of observing j given the
            hidden state i.
        Transition : numpy.ndarray
            Array of shape (N, N) containing transition probabilities.
            Transition[i, j] is the probability of transitioning from
            hidden state i to j.
        Initial : numpy.ndarray
            Array of shape (N, 1) containing the probability of starting
            in a particular hidden state.

        Returns
        -------
        P : float or None
            Likelihood of the observations given the model.
            None on failure.
        B : numpy.ndarray or None
            Array of shape (N, T) containing backward path probabilities.
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j.
            None on failure.
    """
    # reshape initial dist to make compatible with other matrices
    init = Initial.T[0]

    # get shapes
    T = Observation.shape[0]
    N, M = Emission.shape

    # create empty matrix to store backward path probabilities
    B = np.zeros((T, N))

    # set last backward probability with ones
    B[T - 1] = np.ones((N))

    # for each observation, iterating backwards
    for t in range(T - 2, -1, -1):

        # iterate through each state
        for j in range(N):

            # calculate the backward probaility of that state with that
            # observation
            B[t, j] = (B[t + 1] * Emission[:, Observation[t + 1]]
                       ).dot(Transition[j, :])

    # get likelihood of the observations by summing the initial state times
    # the first observation times the backward probabilities from the first
    # timestep
    likelihood = np.sum(init * Emission[:, Observation[0]] * B[0, :])

    # transform to get correct shape and return
    return likelihood, B.T
