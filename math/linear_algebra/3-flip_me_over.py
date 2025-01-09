#!/usr/bin/env python3
"""Transpose matrix."""


def matrix_transpose(matrix):
    """Return the transpose of the given matrix."""
    return [list(row) for row in zip(*matrix)]
