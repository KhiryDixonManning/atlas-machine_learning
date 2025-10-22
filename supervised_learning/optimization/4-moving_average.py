#!/usr/bin/env python3
"""Module containing a function that calculates the weighted moving
   average of a data set."""

import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.

    Parameters:
        data (list): The data to calculate the moving average for.
        beta (float): The weight used for the moving average.

    Returns:
        list: The moving averages of the data.
    """
    m_a = []
    cur_value = 0  # Initialize the first value outside the loop
    for dp in range(len(data)):
        # Update the current value based on the weighted average formula
        cur_value = (beta * cur_value) + (1 - beta) * data[dp]

        # Calculate the bias correction factor
        bias = 1 - (beta ** (dp + 1))

        # Append the current moving average, adjusted for bias
        m_a.append(cur_value / bias)

    return m_a
