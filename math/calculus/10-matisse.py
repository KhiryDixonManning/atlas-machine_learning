#!/usr/bin/env python3

"""
Module: 10-matisse
Description: module defines function to calculate the derivative of polynomial,
where the polynomial is represented as a list of coefficients.
The index of the list corresponds to the power of x for each coefficient.
"""


def poly_derivative(poly):
    """
    Function: poly_derivative
    Description: Computes the derivative of a polynomial represented as a list
    of coefficient. The list index correspond to the power of x, and the value
    at each index is the coefficient.

    Parameters:
    poly (list): A list of numerical coefficients, where the index represents
    the power of x.

    Returns:
    list: A list of coefficient representing the derivative of the polynomial.
    None: If input is invalid (non-list, empty list, or non-numeric elements).

    Notes:
    - If the derivative is zero (the polynomial is constant), the function
    returns [0].
    - The function works for polynomials of any degree, including the constant
    polynomial.
    """
    # Input validation: ensure the input is a list of numbers and not empty
    if not isinstance(poly, list) or len(poly) == 0 or not all(
        isinstance(c, (int, float)) for c in poly
    ):
        return None

    # Special case: If the polynomial has only one coefficient (a constant),
    # the derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative by multiplying each coefficient by its power (index)
    derivative = [
        i * poly[i] for i in range(1, len(poly))
    ]

    # Return the resulting derivative, or [0] if the polynomial was constant
    return derivative if derivative else [0]
