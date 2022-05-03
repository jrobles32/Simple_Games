import pyautogui as py
import cv2 as cv

import time

from Simple_Games.sudoku.img_sudoku.process_img import process_image
from Simple_Games.sudoku.online_sudoku.browser_helpers import sudoku_screenshot


def change_type(chrome_driver=None):
    """
    Takes a user input to determine if a change needs to be made of the input sudoku.

    :param chrome_driver: chrome driver being used for sudoku.
    :return: True if a change of type needs to be made, False otherwise.
    :rtype: bool
    """
    user_cmd = input('Do you want to change the input type? (y/n): ')

    # Validating user input depending on input type.
    if user_cmd.lower() == 'y':

        # Closing chrome driver if user wants to change type.
        if chrome_driver is not None:
            chrome_driver.quit_chrome()

        return True

    elif user_cmd.lower() == 'n':

        # Making sudoku browser active window and clicking on a specific area on the screen to start a new sudoku.
        if chrome_driver is not None:
            titles = py.getAllTitles()
            desired_win = [win_name for win_name in titles if 'sudoku puzzles' in win_name][0]
            sudoku_win = py.getWindowsWithTitle(desired_win)[0]
            sudoku_win.activate()

            py.click(x=2000, y=285)
            py.click(x=1940, y=695)
            time.sleep(2)

        return False

    else:
        print('Not a valid input. Try again.')
        change_type()


def input_photo(input_type, photo_debug=False):
    """
    Determines preprocessing steps of image based on its format type.

    :param input_type: The type of sudoku image one is using. Photo for a camera picture, Screenshot for a screenshot.
        and online for Sudoku image from a website.
    :type input_type: str
    :param photo_debug: True if user wants to see all image processing steps, False otherwise.
    :type photo_debug: bool
    :return: Original Image with solution digits added to appropriate location.
    """
    # Determining the image type.
    if input_type.lower() in ('screenshot', 'photo'):

        # Obtaining image path from user and putting it in a format python can read
        file_loc = input('Input the path of the image: ')
        file_loc.replace('\\', '/')

        # Using CV to read image
        img_file = cv.imread(file_loc)

        # Image preprocessing steps for screenshot/photo and error handling for invalid path.
        try:
            if input_type == 'photo':
                processed_input = process_image(img_file, photo=True, debug=photo_debug)[0]

            else:
                processed_input = process_image(img_file, debug=photo_debug)[0]

            return processed_input

        except AttributeError:
            print('Not a valid file path. Try again.')
            input_photo(input_type)

    elif input_type.lower() == 'online':

        # Obtaining screenshot from browser and processing image.
        sudoku_board = sudoku_screenshot()
        processed_input = process_image(sudoku_board)[1]

        return processed_input

    else:

        print(f'{input_type} is not a valid input. Type "photo", "screenshot", or "online" for input_type.')
        return None


def play_again(chrome_driver=None):
    """
    Takes user input to determine if another sudoku needs to be solved.

    :param chrome_driver: driver being used for web sudoku.
    :return: True if another sudoku needs to be solved, False otherwise.
    :rtype: bool
    """
    # Validating user input.
    while True:
        play = input('\nDo you want to solve another Sudoku? (y/n): ')

        if play.lower() == 'y':
            return True

        elif play.lower() == 'n':

            # Closing web driver if user no longer needs to solve a sudoku online.
            if chrome_driver is not None:
                chrome_driver.quit_chrome()

            return False

        else:
            print('Not a valid input! Type in "y" for yes or "n" for no.')
