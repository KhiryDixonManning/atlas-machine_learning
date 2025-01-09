#!/usr/bin/env python3
"""Matrix addition of arrays."""


def add_arrays(arr1, arr2):
    """Add two arrays element-wise and return the result."""
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
