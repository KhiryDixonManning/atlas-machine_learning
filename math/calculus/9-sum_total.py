#!/usr/bin/env python3

"""
Module: sum_total
Description: This module defines a function to compute
the sum of the squares of the first n integers.
The sum is calculated using a mathematical formula for efficiency:
    sum = n * (n + 1) * (2 * n + 1) / 6
"""

def summation_i_squared(n: int) -> int | None:
    """
    Function: summation_i_squared
    Description: Calculates the sum of squares of the first n positive integers.
    The sum is computed using the formula: n * (n + 1) * (2 * n + 1) / 6

    Parameters:
    n (int): A positive integer specifying the upper limit for the sum of squares.

    Returns:
    int: The sum of the squares of integers from 1 to n.
    None: If the input is not a positive integer.
    """
    # Input validation: Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        return None

    # Direct computation using the sum of squares formula (no loops)
    return (n * (n + 1) * (2 * n + 1)) // 6
