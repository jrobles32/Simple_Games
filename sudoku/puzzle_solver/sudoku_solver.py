import numpy as np


def available_placements(x, y, test_num, board_values):
    """
    Determines if a number between 1 and 9 can be placed at a specific location in the input matrix. In order for number
    to be placed, all Sudoku rules must be met.

    :param x: Desired row index.
    :type x: int
    :param y: Desired column index.
    :type y: int
    :param test_num: Value between 1 and 9 to be tested.
    :type test_num: int
    :param board_values: A 2D Matrix with 9 rows and columns. (Sudoku puzzle)
    :type board_values: ndarray
    :return: True if test number can go into a specific location, False otherwise.
    :rtype: bool
    """
    # Checking if test_num can already be found in desired row.
    for idx in range(9):
        if board_values[x][idx] == test_num:
            return False

    # Checking if test_num can already be found in desired column.
    for idx in range(9):
        if board_values[idx][y] == test_num:
            return False

    # Creating the 3X3 grid that the desired index coordinate would belong to in the Sudoku puzzle.
    box_x = (x // 3) * 3
    box_y = (y // 3) * 3

    # Checking if test_num can already be found in 3X3 grid.
    for row in range(3):
        for col in range(3):
            if board_values[box_x + row][box_y + col] == test_num:
                return False

    return True


def solve_sudoku(sudoku_board):
    """
    Solves a Sudoku puzzle using backtracking. The solution is printed in the terminal.

    :param sudoku_board: 2D array with 9 rows and columns.
    :type sudoku_board: ndarray
    :return: Array of the solved Sudoku puzzle.
    :rtype: ndarray
    """
    # Looping over Sudoku matrix.
    for row_idx in range(9):
        for col_idx in range(9):

            # Control to only change values that are zero.
            if sudoku_board[row_idx][col_idx] == 0:
                for test_value in range(1, 10):

                    # Changing the value of zero to test_value if the value can be placed at the desired location.
                    if available_placements(row_idx, col_idx, test_value, sudoku_board):
                        sudoku_board[row_idx][col_idx] = test_value

                        # Performing recursion to see if test_value can lead to a valid solution.
                        if solve_sudoku(sudoku_board) is not None:
                            return sudoku_board

                        # Changing value back to zero if test_value does not reach a solution.
                        sudoku_board[row_idx][col_idx] = 0

                # Returning none if no valid solution is found.
                return

    print('Solution: \n', np.matrix(sudoku_board))
    return sudoku_board
