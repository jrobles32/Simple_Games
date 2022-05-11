import cv2 as cv

import logging
from configparser import ConfigParser

from online_sudoku.chrome_browser import StartDriver
from online_sudoku.browser_helpers import fill_board
from utils.func_controls import change_type, input_photo, play_again


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


def main():
    # Initalizing debug settings
    config = ConfigParser()
    config.read('sudoku/config.ini')

    debug_setting = config['setting'].getboolean('debug')

    # Allowing user to run program multiple times.
    outer_continue = True
    while outer_continue:

        # Obtaining image input type and establishing control for inner while loop.
        user_format = input("Do you want to solve a Sudoku photo, screenshot, or online? (photo/screenshot/online): ")
        inner_continue = True

        # Input validation for image type that determines what type of image processing will be used.
        if user_format.lower() in ('photo', 'screenshot'):

            # Allowing user to solve the same type of input multiple times.
            while inner_continue:
                solution_img = input_photo(user_format, photo_debug=debug_setting)

                cv.imshow('Sudoku Outline', solution_img)
                cv.waitKey(0)

                cv.destroyAllWindows()

                # Determining if user wants to solve another puzzle and/or change the input type.
                next_game = play_again()

                if next_game:
                    change_format = change_type()
                    inner_continue = False if change_format is True else True

                else:
                    print('Next time, try to solve the puzzle yourself!!!')
                    outer_continue = inner_continue = False

        elif user_format.lower() == "online":
            logger.info('Initializing web sudoku')

            # Setting default website that will be used and starting web driver
            url = 'https://sudoku.com/evil/'
            sud_web = StartDriver().desired_site(url)

            # Allowing user to solve multiple online puzzles.
            while inner_continue:

                # Obtaining solution results to input into online puzzle.
                final_nums = input_photo(user_format)
                fill_board(final_nums)

                # Determining if user wants to solve another puzzle and/or change the input type.
                next_game = play_again(chrome_driver=sud_web)

                if next_game:
                    change_format = change_type(chrome_driver=sud_web)
                    inner_continue = False if change_format is True else True

                else:
                    print('Next time, try to solve the puzzle yourself!!!')
                    outer_continue = inner_continue = False

        else:
            logger.warning(f'{user_format} is not a valid input. Type "photo", "screenshot", or "online".\n')

if __name__ == '__main__':
    main()
