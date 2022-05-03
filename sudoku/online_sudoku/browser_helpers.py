import cv2 as cv
import numpy as np
import pyautogui as py

import time


def fill_board(solution_values):
    """
    Inputs the solution to the sudoku puzzle in the Chrome web driver.

    :param solution_values: The values to be inputted into each box of the sudoku
    :type solution_values: ndarray
    :return: None
    """
    print('\nFilling Board. Do not engage with keyboard!!!')
    # Selecting the Chrome driver window and making it the active window
    titles = py.getAllTitles()
    desired_win = [win_name for win_name in titles if 'sudoku puzzles' in win_name][0]
    sudoku_win = py.getWindowsWithTitle(desired_win)[0]
    sudoku_win.activate()

    count = []
    for value in solution_values:
        # Inputting solution value into box, moving to the right once, and appending a token to list.
        py.press(str(value))
        py.press('right')
        count.append(0)

        # Once token count has reached 81 all sudoku boxes are filled
        if len(count) == 81:
            break
        # If token count has a remainder of 0, moving down and to the farthest left box to begin and fill next row.
        elif len(count) % 9 == 0:
            py.press('down')
            py.press('left', presses=8)


def sudoku_screenshot():
    """
    Takes a screenshot of the specific area the sudoku is in the Chrome web driver.

    :return: The cv ready image of the sudoku
    """
    # Selecting Chrome driver window.
    titles = py.getAllTitles()
    desired_win = [win_name for win_name in titles if 'sudoku puzzles' in win_name][0]
    sudoku_win = py.getWindowsWithTitle(desired_win)[0]

    # Resizing window, moving it, and ensuring that it is the active window.
    sudoku_win.resizeTo(1294, 1000)
    sudoku_win.moveTo(1273, 0)
    sudoku_win.activate()

    # Clicking the top left most box of the sudoku
    time.sleep(2)
    py.click(x=1350, y=285)
    time.sleep(1)

    # Creating the screenshot and converting colors to be ready for CV preprocessing
    sudoku_img = py.screenshot(region=(1320, 255, 509, 508))
    cv_sudoku = cv.cvtColor(np.array(sudoku_img), cv.COLOR_RGB2BGR)

    return cv_sudoku
