from online_sudoku.chrome_browser import StartDriver
from online_sudoku.browser_helpers import fill_board
from utils.func_controls import change_type, input_photo, play_again


def main():
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
                input_photo(user_format, photo_debug=True)

                # Determining if user wants to solve another puzzle and/or change the input type.
                next_game = play_again()

                if next_game:
                    change_format = change_type()
                    inner_continue = False if change_format is True else True

                else:
                    print('Next time, try to solve the puzzle yourself!!!')
                    outer_continue = inner_continue = False

        elif user_format.lower() == "online":
            print('Initializing web sudoku....')

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
            print('Not a valid input. Type "photo" if sudoku is a file or type "web" to solve online sudoku.')


if __name__ == '__main__':
    main()
