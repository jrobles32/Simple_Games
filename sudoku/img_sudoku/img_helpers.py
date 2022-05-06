import cv2 as cv
import numpy as np

import os

from .img_prep import prep_box

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


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
    model = tf.keras.models.load_model('sudoku/utils/new_number_reader.model')
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
