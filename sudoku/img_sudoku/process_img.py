from .img_helpers import *
from .img_prep import largest_contour, corrected_points, bounding_boxes
from sudoku.puzzle_solver.sudoku_solver import solve_sudoku


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
    print('\nProcessing image....')

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

            cv.destroyAllWindows()

        return paste_results, solved_board

    else:
        print('An error with the image occurred. No contours were found.')
