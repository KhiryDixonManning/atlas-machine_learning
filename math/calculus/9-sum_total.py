#!/usr/bin/env python3

"""
Module: sum_total
Description: This module defines a function to compute
the sum of squares of the first n integers using the formula:
    sum = n * (n + 1) * (2 * n + 1) / 6
"""

from typing import Union

def summation_i_squared(n: int) -> Union[int, None]:
    """
    Description: Calculates the sum of squares of the first n positive integers.
    Uses the formula: n * (n + 1) * (2 * n + 1) / 6

    Parameters:
    n (int): A positive integer specifying the upper limit.

    Returns:
    int: The sum of squares from 1 to n.
    None: If the input is invalid.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
