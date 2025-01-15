#!/usr/bin/env python3

"""
Module: 17-integrate
Description: This module defines a function to compute the indefinite integral of a polynomial,
where the polynomial is represented by a list of coefficients. The function returns the coefficients of
the integral, including an optional constant of integration.
"""


def poly_integral(poly, C=0):
    """
    Function: poly_integral
    Description: Computes the indefinite integral of a polynomial represented by a list of coefficients.
    The index of the list corresponds to the power of x, and the value at each index is the coefficient.

    Parameters:
    poly (list): A list of coefficients representing the polynomial.
                 Example: [5, 3, 0, 1] represents f(x) = x^3 + 3x + 5.
    C (int, float): The constant of integration (default is 0).

    Returns:
    list: A list of coefficients representing the integral of the polynomial.
    None: If the input is invalid (non-list, empty list, or non-numeric coefficients).

    Notes:
    - The function handles polynomials of any degree and includes the constant of integration.
    - Non-integer results are kept as floats, while integer results are cast to `int`.
    - If the polynomial consists solely of zeros, the function returns the constant of integration alone.
    """
    # Input validation
    if not isinstance(poly, list) or len(poly) < 1:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    # If the polynomial is just [0], return the constant
    if poly == [0]:
        return [C]

    # Insert the constant of integration at the start of the list
    integral = [C]

    # Compute the integral by dividing each coefficient by its power (index + 1)
    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        integral.append(int(new_coef) if new_coef.is_integer() else new_coef)

    return integral