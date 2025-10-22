#!/usr/bin/env python3
""" module that contains function that finds determinate of square matrix """

# [[a b c]
#  [e f g]
#  [h i j]]

# (a*((f * j) - (g * i))) - (b*((e * j) - (g * h))) + (c*((e * i) - (f * h)))


def submatrix(matrix, col):
    """ function that finds the submatrix for a given value in the first row
    of a 3x3 matrix

    Args:
        matrix (list of lists): A square matrix represented as a list of lists.
        col (int): current column positon (the row is always 0)

    Returns:
        list of lists: the submatrix for the given value
    """
    row = []
    submatrix = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != 0 and j != col:
                row.append(matrix[i][j])
        if i != 0:
            submatrix.append(row)
            row = []
    return submatrix


def determinant(matrix):
    """
    function that calculate the determinant of a square matrix.

    Args:
        matrix (list of lists): A square matrix represented as a list of lists.

    Returns:
        float: The determinant of the input matrix.
    """
    # initial check if matrix is list of lists
    if not isinstance(matrix, list) or len(
            matrix) == 0 or not isinstance(matrix[0], list):
        raise TypeError('matrix must be a list of lists')

    # if matrix is a list containing at least one list at the beginning, check
    # to make sure list is not empty. if so, the determinate is
    if len(matrix[0]) == 0:
        return 1

    # if matrix is a list containing a list with only 1 value, the determinate
    # is that value
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # check to see if matrix has same length as first list inside of it
    if len(matrix) == len(matrix[0]):

        # if they are the same length, iterate through each nested list and
        # once again check type and length to be sure it is a list and the same
        # length as the outer list
        for x in range(len(matrix)):
            if not isinstance(matrix[x], list):
                raise TypeError('matrix must be a list of lists')
            if len(matrix[x]) != len(matrix):
                raise ValueError('matrix must be a square matrix')

        # if length of matrix is 2, calculate the determinate
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        # if length is anything above 2, find each submatrix and recursively
        # call this function with said submatrix until the submatrix reaches
        # dims of 2x2, gradually calculating the determinate along the way
        else:
            det = 0
            for i in range(len(matrix)):
                sm = submatrix(matrix, i)
                det = det + ((-1) ** i) * matrix[0][i] * determinant(sm)
            return det

    # once again raise error if all of the above didnt apply to matrix
    else:
        raise ValueError('matrix must be a square matrix')
