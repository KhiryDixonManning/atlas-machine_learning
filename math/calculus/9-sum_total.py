#!/usr/bin/env python3

"""
This module contains the function summation_i_squared(n)
"""

def summation_i_squared(n):
    """
    :arg n: upper limit
    :return: summation of i squared
    """
    if not isinstance(n, int) or n == 0 or n < 1:
        return None  # Explicit return of None for invalid input
    return (n * (n + 1) * (2 * n + 1)) // 6
