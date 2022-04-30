import pyautogui as py
import time
import cv2 as cv
import numpy as np
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


class StartDriver:
    """
    An object that represents a Chrome browser.
    """

    def __init__(self):
        """
        Establishing the path of browser and adding extensions
        """
        chrome_path = 'D:/Py_ChromeDriver/chromedriver.exe'
        ad_ex = 'C:/Users/Javier/AppData/Local/Google/Chrome/User ' \
                'Data/Default/Extensions/cjpalhdlnbpafiamejdnhcphjbkeiagm/1.42.4_0 '

        # Adding ad blocker and starting up browser
        chrome_options = Options()
        chrome_options.add_argument('load-extension=' + ad_ex)
        self.driver = webdriver.Chrome(chrome_path, options=chrome_options)

    def desired_site(self, site_url):
        """
        Takes the object to a new/another website.

        :param site_url: website user wants to visit
        :type site_url: str
        :return: updated web location
        """
        self.driver.get(site_url)
        return self

    def quit_chrome(self):
        """
        Exits the web browser.

        :return: None
        """
        self.driver.quit()


def display_numbers(image, numbers, color=(0, 255, 0)):
    """
    Places a digit in a specific area of an image. Values of zero will be ignored and given an empty space.

    :param image: Desired image to overlay digits on.
    :type image: src
    :param numbers: Digits to be displayed on image.
    :type numbers: list
    :param color: Decimal code for desired color of digits to be displayed. Default color is green.
    :return: Original image with desired digits put on image.
    """
    # Creating the dimensions that each digit will take up in the image.
    sec_w = int(image.shape[1] / 9)
    sec_h = int(image.shape[0] / 9)

    # Looping over Sudoku matrix values
    for row_idx in range(0, 9):
        for col_idx in range(0, 9):
            # Controlling so iterations skip index values that contain zero.
            if numbers[(col_idx * 9) + row_idx] != 0:
                # Putting digit on section of image.
                cv.putText(img=image,
                           text=str(numbers[(col_idx * 9) + row_idx]),
                           org=(row_idx * sec_w + int(sec_w / 2) - 10, int((col_idx + 0.8) * sec_h)),
                           fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,
                           fontScale=2,
                           color=color,
                           thickness=2,
                           lineType=cv.LINE_AA)
    return image


def prep_box(box, photo=False):
    """
    Creates a mask of an image that removes outer boundaries. Also, resizes the image and puts it into right shape for
    digit recognition model. If it is a camera image it preprocess the image.

    :param box: Path of an image.
    :type box: src
    :param photo: True if image is a photo, false otherwise.
    :type photo: bool
    :return: A resize and reshaped image
    """
    # Area of image to isolate
    x = y = 5
    w = h = 40

    # Creating a black image (mask) with same dimension as box and placing isolated area inside mask
    mask = np.zeros(box.shape, np.uint8)
    mask[y:y + h, x:x + w] = box[y:y + h, x:x + w]

    # Converting image to an array and cropping the image to better center the digit
    img_box = np.asarray(mask)
    img_box = img_box[4:img_box.shape[0] - 4, 4: img_box.shape[1] - 4]

    if photo:
        # Preprocessing to enhance isolation of digit in image
        blur = cv.GaussianBlur(img_box, (13, 13), 0)
        img_box = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)[1]

    # Resizing image and normalizing it
    img_box = cv.resize(img_box, (28, 28))
    norm_img = img_box / 255

    # Reshaping image to fit model
    final_img = norm_img.reshape(1, 28, 28, 1)

    return final_img


def stack_images(scale, img_array):
    """
    Takes in multiple images and stacks them horizontally and vertically to create a new image. If images do not share
    the same dimensions, images are scaled to the dimensions of the first image in the inputted tuple.

    :param scale: Value to enlarge or reduce input images.
    :type scale: float
    :param img_array: Tuple of lists containing desired images. The number of lists inside tuple represent how many rows
        of images the final output will have. The number of images inside a list represent the number of columns. All
        lists must be of equal length. Input an empty image if length are not the same.
    :type img_array: any
    :return: A new image composed of all the images in img_array.
    """
    # Finding the total number of rows and columns that will make up the final image.
    rows = len(img_array)
    cols = len(img_array[0])

    # Determining if first variable in tuple is a list or just an image.
    rows_available = isinstance(img_array[0], list)

    # Finding the dimension of the first image in the inputted tuple.
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    # Control that determines if final image will have multiple rows or just one.
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):

                # Converting dimensions of current image when they do not match the first. And, applying desired scale.
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                dsize=(0, 0),                           # Can be changed to None
                                                fx=scale,
                                                fy=scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                dsize=(img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                fx=scale,
                                                fy=scale)

                # Converting black and white images to ensure they contain three color channels.
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)

        # Creating a list of black images that length is equal to the user desired number of rows.
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows

        # Replacing black images with a horizontal stack of the images in an element of the inputted tuple.
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])

        # Vertically stacking the horizontal stacks of images previously created to develop one image.
        ver = np.vstack(hor)

    else:
        for x in range(0, rows):

            # Converting dimensions of current image when they do not match the first. And, applying desired scale.
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x],
                                         dsize=(0, 0),
                                         fx=scale,
                                         fy=scale)
            else:
                img_array[x] = cv.resize(img_array[x],
                                         dsize=(img_array[0].shape[1], img_array[0].shape[0]),
                                         fx=scale,
                                         fy=scale)

            # Converting black and white images to ensure they contain three color channels.
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)

        # Horizontally stacking images in the inputted tuple.
        hor = np.hstack(img_array)
        ver = hor

    return ver


def corrected_points(points):
    """
    Serves to reorganize the input array of a four-sided figure to enable one to do a perspective transformation.

    :param points: Vertices of a four sided shape.
    :type points: ndarray
    :return: A 3D array with vertex values ordered top left, top right, bottom left, bottom right.
    """
    # Creating a 3D array with values set to zero.
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    # Reshaping input 3D array to 2D.
    points = points.reshape(4, 2)

    # Adding together the values in each row.
    list_add = points.sum(1)

    # Setting first value in the new 3D array to the min of the sums and the last to the max of add.
    new_points[0] = points[np.argmin(list_add)]
    new_points[3] = points[np.argmax(list_add)]

    # Subtracting the second value in a row by the first.
    list_diff = np.diff(points, axis=1)

    # Setting second value of new 3D array to min positive value and the third to the max value from diff.
    new_points[1] = points[np.argmin(list_diff)]
    new_points[2] = points[np.argmax(list_diff)]

    return new_points


def largest_contour(contours):
    """
    Uses a list of contours to determine if a square figure can be created from the vertices. If one is created the area
    of the largest square found and its vertex points are taken.

    :param contours: Detected contours from an image.
    :type contours: list
    :return: tuple (largest, max_area)
        WHERE
        float largest: Vertex points of the square.
        float max_area: Area inside the square.
    """
    # Initializing empty variables to store final results
    largest = np.array([])
    max_area = 0

    for cont in contours:
        # Finding area of contours and eliminating small values.
        area = cv.contourArea(cont)
        if area > 50:
            # Determining perimeter of Contour and stating it is closed.
            peri = cv.arcLength(cont, True)
            # Creating an approximation of the shape created from the contour
            approx = cv.approxPolyDP(cont, 0.02 * peri, True)

            # Selecting vertices that form a shape with four sides and setting max values
            if area > max_area and len(approx) == 4:
                largest = approx
                max_area = area

    return largest, max_area


def bounding_boxes(image):
    """
    Splits an inputted image into multiple boxes.

    :param image: path of image to be split
    :type image: src
    :return: 81 new images
    :rtype: list
    """
    # Creating 9 vertical splits
    img_rows = np.vsplit(image, 9)

    # Empty list to store images
    boxes = []

    for row in img_rows:
        # Creating 9 horizontal splits
        cols = np.hsplit(row, 9)
        for box in cols:
            # Placing new images into initialized list
            boxes.append(box)
    return boxes


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


def modified_array(nums_scanned, solution_nums):
    """
    Determines if a value is shared between two lists at a specific index value. Values that are shared are replaced
    with zero.

    :param nums_scanned: Sudoku puzzle numbers with zero representing empty spaces.
    :type nums_scanned: ndarray
    :param solution_nums: All the numbers for the solution of Sudoku puzzle. No zeros.
    :type solution_nums: ndarray
    :return: Only the solution values of the Sudoku puzzle
    :rtype: list
    """
    # Initializing empty list to store values.
    not_shared = []

    for idx, value in enumerate(solution_nums):

        # Adding zero if values match. If they do not, adding the value in solution_nums to not_shared.
        if nums_scanned[idx] == solution_nums[idx]:
            not_shared.append(0)
        else:
            not_shared.append(value)

    return not_shared


def process_image(image, img_width=504, img_height=504, photo=False, debug=False):
    """
    Applies image processing steps to prepare sudoku image for the application of a trained model based on mnist
    dataset. The results are used to determine a solution to the puzzle.

    :param image: CV loaded image
    :param img_width: Desired width of the image of the sudoku box.
    :type img_width: int
    :param img_height: Desired height of the image of the sudoku box.
    :type img_height: int
    :param photo: True if user is using photo, False otherwise.
    :type photo: bool
    :param debug: True if user wants to see image processing steps, False otherwise.
    :type debug: bool
    :return: Original image with solution pasted into boxes and list of values found.
    """
    print('Processing image....')

    # Converting img to grayscale and applying a slight blur to better detect edges
    img_gray = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    # Converting pixels based on intensity and inverting the colors of the image
    thresh = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    inverted = cv.bitwise_not(thresh)

    # Finding all external contours
    cont = cv.findContours(inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]

    # Creating an image with external contours, and selecting the largest contour
    img_cont = cv.drawContours(image.copy(), cont, -1, (0, 255, 0), 5)
    lg_contour, max_area = largest_contour(cont)

    if lg_contour.size != 0:
        # Creating image of Sudoku boundary
        puzzle = cv.drawContours(image.copy(), lg_contour, -1, (0, 0, 255), 20)

        # Correctly setting up the coordinates of lg_contour for transformation and creating desired output shape
        fix_points = corrected_points(lg_contour)
        pts1 = np.float32(fix_points)
        pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])

        # Using transformation matrix to create a warp perspective image.
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        img_warp = cv.warpPerspective(image.copy(), matrix, (img_width, img_height))

        # Converting created image to black and white, changing pixels based on intensity, and inverting the colors.
        warp_bw = cv.cvtColor(img_warp, cv.COLOR_BGR2GRAY)
        warp_thresh = cv.adaptiveThreshold(warp_bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        warp_inverted = cv.bitwise_not(warp_thresh)

        # Isolating each individual digit and applying digit recognition model.
        boxes = bounding_boxes(warp_inverted)
        if photo:
            nums_found = digit_recognition(boxes, photo=True)
        else:
            nums_found = digit_recognition(boxes)

        # Creating blank image.
        img_blank = np.zeros([img_width, img_height, 3], dtype=np.uint8)

        # Applying puzzle numbers recognized to a blank image.
        display_num = img_blank.copy()
        img_detected = display_numbers(display_num, nums_found, color=(0, 0, 255))

        # Spitting list with 81 values into 9 different arrays and attempting to solve Sudoku.
        print('Beginning to solve the puzzle....')
        board = np.array_split(nums_found, 9)
        solve_sudoku(board)

        # Flattening the 9 arrays back to one and removing shared values between solution and puzzle numbers.
        solved_board = np.array(board).flatten()
        solved_digits = modified_array(nums_found, solved_board)

        # Applying digits of the solution to a blank image.
        new_digits = img_blank.copy()
        digits_img = display_numbers(new_digits, solved_digits)

        # Posting puzzle solution over original image.
        paste_results = cv.addWeighted(digits_img, 1, img_warp.copy(), 0.5, 1)

        if debug:
            # Creating an array of images in different steps of puzzle solution to find where an error has occurred.
            img_array = ([image, inverted, img_cont, puzzle], [img_warp, img_detected, digits_img, paste_results])
            img_stack = stack_images(0.5, img_array)

            cv.imshow('Sudoku Outline', img_stack)
            cv.waitKey(0)

        return paste_results, solved_board

    else:
        print('An error with the image occurred. No contours were found.')


def digit_recognition(boxes, photo=False):
    """
    Recognizes the value of a digit inside a bounding box. Digit recognition model uses the MNIST dataset with added
    personal characters. Empty images are given a value of zero.

    :param boxes: Images that have a digit centralized.
    :param photo: True is original image is a photo, False otherwise.
    :type photo: bool
    :return: 81 values recognized in image.
    :rtype: list
    """
    # Loading digit recognition model and creating an empty list to store numbers found
    model = tf.keras.models.load_model('D:/Trained_Models/new_number_reader.model')
    nums_found = []

    for box in boxes:
        # Centering digits in bounding boxes and changing array format.
        if photo:
            box_prep = prep_box(box, photo=True)
        else:
            box_prep = prep_box(box)

        # Applying digit recognition model to image.
        prediction = model.predict(box_prep)

        # Selecting the digit with the highest probability and the probability value.
        class_idx = np.argmax(prediction, axis=-1)
        probability = np.amax(prediction)

        # Establishing minimum probability value for digit to get appended. Appending zero if value is not reached.
        if probability > 0.70:
            nums_found.append(class_idx[0])
        else:
            nums_found.append(0)

    return nums_found


def fill_board(solution_values):
    """
    Inputs the solution to the sudoku puzzle in the Chrome web driver.

    :param solution_values: The values to be inputted into each box of the sudoku
    :type solution_values: ndarray
    :return: None
    """
    print('\nFilling Board. Do not engage with keyboard!!!\n')
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


def play_again(chrome_driver=None, image=False):
    """
    Takes user input to determine if another sudoku needs to be solved. Can only use one of variables.

    :param chrome_driver: driver being used for web sudoku.
    :param image: True if input is a screenshot/photo, False otherwise.
    :type image: bool
    :return: True if another sudoku needs to be solved, False otherwise.
    :rtype: bool
    """
    if image:
        # Validating user input for an image.
        while True:
            play = input('\nDo you want to solve another Sudoku? (y/n): ')
            if play.lower() == 'y':
                return True
            elif play.lower() == 'n':
                return False
            else:
                print('Not a valid input! Type in "y" for yes or "n" for no.')
    else:
        # Validating user input for a web sudoku.
        while True:
            play = input('Do you want to solve another Sudoku? (y/n): ')
            if play.lower() == 'y':
                return True

            # Closing web driver if user no longer needs to solve a sudoku online.
            elif play.lower() == 'n':
                chrome_driver.quit_chrome()
                return False

            else:
                print('Not a valid input! Type in "y" for yes or "n" for no.')


def image_type():
    """
    Takes a user input to determine if input is a screenshot or a photo. Important to distinguish since they need
    different preprocessing steps.

    :return: True if input is a photo, False if a screenshot.
    :rtype: bool
    """
    img_form = input('Is the image a photo or a screenshot? (photo/screen): ')

    # Validating user input
    if img_form.lower() == 'photo':
        return True
    elif img_form.lower() == 'screen':
        return False
    else:
        print('Not a valid input. Type "photo" or "screen".')
        image_type()


def input_photo(photo_debug=False):
    """
    Determines preprocessing steps of image based on if the image is a photo or screenshot.

    :param photo_debug: True if user wants to see all image processing steps, False otherwise
    :type photo_debug: bool
    :return: Original Image with solution digits added to appropriate location.
    """
    # Determining image type
    img_format = image_type()

    # Obtaining image path from user and putting it in a format python can read
    file_loc = input('Input the path of the image: ')
    file_loc.replace('\\', '/')

    # Using CV to read image
    img_file = cv.imread(file_loc)

    # Image preprocessing steps for photo and error handling for invalid path
    if img_format:
        try:
            processed_input = process_image(img_file, photo=img_format, debug=photo_debug)[0]
            return processed_input

        except AttributeError:
            print('Not a valid file path. Try again.')
            input_photo()

    # Image preprocessing steps for screenshot and error handling for invalid path
    else:
        try:
            processed_input = process_image(img_file, debug=photo_debug)[0]
            return processed_input

        except AttributeError:
            print('Not a valid file path. Try again.')
            input_photo()


def change_type(chrome_driver=None, image=False):
    """
    Takes a user input to determine if a change of sudoku type needs to be made. Only one of the variables should be
    used.

    :param chrome_driver: chrome driver being used for sudoku.
    :param image: True is one is using an image, False otherwise
    :type image: bool
    :return: True if a change of type needs to be made, False otherwise.
    :rtype: bool
    """
    user_cmd = input('Do you want to change the input type? (y/n): ')

    # Validating user input depending on input type.
    if image:
        if user_cmd.lower() == 'y':
            return True
        elif user_cmd.lower() == 'n':
            return False
        else:
            print('Not a valid input. Try again.')
            change_type()
    else:
        # Closing chrome driver if user wants to change type
        if user_cmd.lower() == 'y':
            chrome_driver.quit_chrome()
            return True
        # Clicking on a specific area on the screen to start a new sudoku in the chrome driver
        elif user_cmd.lower() == 'n':
            py.click(x=2000, y=285)
            py.click(x=1940, y=695)
            time.sleep(2)
            return False
        else:
            print('Not a valid input. Try again.')
            change_type()


def main():
    user_format = input("Do you want to solve a sudoku image or online sudoku? (image/web): ")
    if user_format.lower() == "web":
        print('Initializing web sudoku....')
        url = 'https://sudoku.com/evil/'
        sud_web = StartDriver().desired_site(url)
        done = False
        while not done:
            sudoku_board = sudoku_screenshot()
            final_nums = process_image(sudoku_board)[1]
            fill_board(final_nums)
            next_game = play_again(chrome_driver=sud_web)
            if next_game:
                sudoku_format = change_type(chrome_driver=sud_web)
                if sudoku_format:
                    main()
                else:
                    continue
            else:
                print('Next time, try to solve the puzzle yourself!!!')
                done = True

    elif user_format.lower() == 'image':
        done = False
        while not done:
            input_photo(photo_debug=True)
            cv.destroyAllWindows()
            next_game = play_again(image=True)
            if next_game:
                sudoku_format = change_type(image=True)
                if sudoku_format:
                    main()
                else:
                    continue
            else:
                print('Next time, try to solve the puzzle yourself!!!')
                done = True

    else:
        print('Not a valid input. Type "photo" if sudoku is a file or type "web" to solve online sudoku.')
        main()


if __name__ == '__main__':
    main()
