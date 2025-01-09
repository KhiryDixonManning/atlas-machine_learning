#!/usr/bin/env python3
"""Write a function that calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Return the shape (dimensions) of the given matrix."""
    shape = []  # Initialize empty list to store the dimensions of the matrix.

    while isinstance(matrix, list):  # Continue if the matrix is a list
        shape.append(len(matrix))  # Add length from current dimension to shape
        matrix = matrix[
            0] if matrix else []  # Move first element of dimension if no stop
    return shape
