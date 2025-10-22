#!/usr/bin/env python3
""" module containing function that calculates the adjugate matrix of a
    given square matrix."""


def submatrix(matrix, x, y):
    """ function that finds the submatrix for a given value in the first row
    of a 3x3 matrix

    Args:
        matrix (list of lists): A square matrix represented as a list of lists.
        x (int): current row positon
        y (int): current column positon

    Returns:
        list of lists: the submatrix for the given value
    """
    row = []
    submatrix = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != x and j != y:
                row.append(matrix[i][j])
        if i != x:
            submatrix.append(row)
            row = []
    return submatrix


def determinate(matrix):
    """
    function that calculate the determinant of a square matrix.

    Args:
        matrix (list of lists): A square matrix represented as a list of lists.

    Returns:
        int: The determinant of the input matrix.
    """
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(len(matrix)):
        det += ((-1) ** j) * matrix[0][j] * \
            determinate(submatrix(matrix, 0, j))
    return int(det)


def adjugate(matrix):
    """
    function that calculates the adjugate matrix of a given square matrix.

    Args:
        matrix (list of lists): A square matrix whose adjugate matrix is to
                                be calculated.

    Returns:
        list of lists: The adjugate matrix of the input matrix.
    """

    # first store length of outer list
    length = len(matrix)

    # if length is 0, raise value error indicating the list does not contain
    # other lists
    if length == 0:
        raise TypeError('matrix must be a list of lists')

    # if length is greater than 0, check all types and lengths of inner items
    else:
        for i in range(length):
            if len(matrix[i]) != length:
                raise ValueError('matrix must be a non-empty square matrix')
            elif not isinstance(matrix[i], list):
                raise TypeError('matrix must be a list of lists')

            # if both inner and outer list have length of 1, return a list
            # containing 1 in a list
            elif len(matrix[i]) == 1 and length == 1:
                return [[1]]

    # make copy of matrix
    mnr = [x[:] for x in matrix]

    # if both lengths are 2, switch positions of values to find minor of 2x2
    # matrix
    if len(matrix) == 2:
        for i in range(length):
            for j in range(length):
                for x in range(length):
                    for y in range(length):
                        if x != i and y != j:
                            mnr[i][j] = matrix[x][y]
                if (i == 0 and j == 1) or (i == 1 and j == 0):
                    mnr[i][j] *= -1

    # if matrix is non-empty, square, and a list of lists and have a length
    # greater that 2
    else:
        pos = 0
        # iterate through each value in the matrix
        for i in range(length):
            for j in range(length):

                # find the submatricies of each value
                sm = submatrix(matrix, i, j)

                # recursively calculate the matrix of minors using determinate
                # function
                mnr[i][j] = determinate(sm)

                # if length is any other side, multiple every other value
                # by -1
                if pos > 0 and pos % 2 != 0:
                    mnr[i][j] *= -1
                pos += 1

    res = []
    for i in range(len(matrix[0])):
        col = []
        for row in mnr:
            col.append(row[i])
        res.append(col)

    return res
