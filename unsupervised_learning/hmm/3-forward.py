#!/usr/bin/env python3
""" module containing function that performs the forward algorithm for a
    Hidden Markov Model."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ function that performs the forward algorithm for a Hidden Markov
        Model.

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
        F : numpy.ndarray or None
            Array of shape (N, T) containing forward path probabilities.
            F[i, j] is the probability of being in hidden state i at
            time j given the previous observations.
            None on failure.
    """

    # reshape initial dist to make compatible with other matrices
    init = Initial.T[0]

    # get shapes
    T = Observation.shape[0]
    N, M = Emission.shape

    # create empty matrix to store forward path probabilities
    F = np.zeros((T, N))

    # get first forward probability with probability of the initial state
    # times the emission probability of first observation
    F[0, :] = init * Emission[:, Observation[0]]

    # for each observation
    for t in range(1, T):

        # iterate through each state
        for j in range(N):

            # calculate the forward probaility of that state with that
            # observation
            F[t, j] = F[t - 1].dot(Transition[:, j]) * \
                Emission[j, Observation[t]]

    # get likelihood of the observations by summing the forward probabilities
    # from the last timestep
    likelihood = np.sum(F[T - 1])

    # transform to get correct shape and return
    return likelihood, F.T
