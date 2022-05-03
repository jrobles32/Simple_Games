import cv2 as cv
import numpy as np


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
