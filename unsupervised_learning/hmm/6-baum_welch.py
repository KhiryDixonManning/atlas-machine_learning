#!/usr/bin/env python3
""" tbc """
import numpy as np


def forward(Observation, Transition, Emission, Initial):
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
    return likelihood, F


def backward(Observation, Transition, Emission, Initial):
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
    return likelihood, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ tbc """

    N, M = Emission.shape
    T = Observations.shape[0]

    if iterations > T:
        iterations = T + 15

    for _ in range(iterations):

        l_f, F = forward(Observations, Transition, Emission, Initial)
        l_b, B = backward(Observations, Transition, Emission, Initial)

        delta = np.zeros((N, N, T - 1))

        for t in range(T - 1):

            denom = np.dot(np.dot(F[t, :].T, Transition) *
                           Emission[:, Observations[t + 1]].T, B[t + 1, :])

            for i in range(N):
                numer = F[t, i] * Transition[i, :] * \
                    Emission[:, Observations[t + 1]].T * B[t + 1, :].T

                delta[i, :, t] = numer / denom

        gamma = np.sum(delta, axis=1)

        Transition = np.sum(delta, axis=2) / \
            np.sum(gamma, axis=1)[:, np.newaxis]

        gamma = np.column_stack(
            (gamma, np.sum(delta[:, :, T - 2], axis=0)[:, np.newaxis]))

        denom = np.sum(gamma, axis=1)[:, np.newaxis]

        for i in range(M):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)

        Emission /= denom

    return Transition, Emission
