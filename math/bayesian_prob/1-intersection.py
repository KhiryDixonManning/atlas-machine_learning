#!/usr/bin/env python3
""" tbw """
import numpy as np


def intersection(x, n, P, Pr):
    """ tbw """

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, type(np.array([]))) or len(
            P.shape) < 1 or P.shape[0] <= 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, type(np.array([]))) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    temp = []

    for i in range(len(P)):

        if P[i] > 1 or P[i] < 0:
            raise ValueError('All values in P must be in the range [0, 1]')

        if Pr[i] > 1 or Pr[i] < 0:
            raise ValueError('All values in Pr must be in the range [0, 1]')

        n_f = 1
        for j in range(1, n + 1):
            n_f *= j
        x_f = 1
        nx_f = 1
        for j in range(1, x + 1):
            x_f *= j
        for j in range(1, (n - x) + 1):
            nx_f *= j
        coeff = n_f / (x_f * nx_f)
        Lp = coeff * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        ints = Lp * Pr[i]
        temp.append(ints)

    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    return np.array(temp)
