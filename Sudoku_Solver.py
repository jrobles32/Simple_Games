import pyautogui as py
import time
import cv2 as cv
import numpy as np
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

##############################################################################################
path = r'C:\Users\Javier\OneDrive\Pictures\Screenshots\Sudoku_Test6.png'.replace('\\', '/')
model = tf.keras.models.load_model('new_number_reader.model')

img_width = 504
img_height = 504

img_blank = np.zeros([img_width, img_height, 3], dtype=np.uint8)
##############################################################################################


class StartDriver:
    def __init__(self):
        chrome_path = 'C:/Program Files/Python39/Lib/site-packages/helium/_impl/webdrivers/windows/chromedriver.exe'
        ad_ex = 'C:/Users/Javier/AppData/Local/Google/Chrome/User ' \
                'Data/Default/Extensions/cjpalhdlnbpafiamejdnhcphjbkeiagm/1.37.2_0 '
        chrome_options = Options()
        chrome_options.add_argument('load-extension=' + ad_ex)
        self.driver = webdriver.Chrome(chrome_path, options=chrome_options)

    def desired_site(self, site_url):
        self.driver.get(site_url)
        return self

    def quit_chrome(self):
        self.driver.quit()


def display_numbers(image, numbers, color=(0, 255, 0)):
    sec_w = int(image.shape[1] / 9)
    sec_h = int(image.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv.putText(image, str(numbers[(y * 9) + x]),
                           (x * sec_w + int(sec_w / 2) - 10, int((y + 0.8) * sec_h)),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv.LINE_AA)
    return image


def prep_box(box, photo=False):
    x = y = 5
    w = h = 40

    mask = np.zeros(box.shape, np.uint8)
    mask[y:y + h, x:x + w] = box[y:y + h, x:x + w]

    img_box = np.asarray(mask)
    img_box = img_box[4:img_box.shape[0] - 4, 4: img_box.shape[1] - 4]

    if photo:
        blur = cv.GaussianBlur(img_box, (13, 13), 0)
        img_box = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)[1]

    img_box = cv.resize(img_box, (28, 28))

    norm_img = img_box / 255
    final_img = norm_img.reshape(1, 28, 28, 1)
    return final_img


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                         scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def corrected_points(points):
    points = points.reshape(4, 2)
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


def largest_contour(contours):
    largest = np.array([])
    max_area = 0
    for cont in contours:
        area = cv.contourArea(cont)
        if area > 50:
            peri = cv.arcLength(cont, True)
            approx = cv.approxPolyDP(cont, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                largest = approx
                max_area = area
    return largest, max_area


def finding_digits(image):
    img_rows = np.vsplit(image, 9)

    boxes = []
    for row in img_rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def available_placements(x, y, test_num, board_values):
    for idx in range(9):
        if board_values[x][idx] == test_num:
            return False
    for idx in range(9):
        if board_values[idx][y] == test_num:
            return False

    box_x = (x // 3) * 3
    box_y = (y // 3) * 3
    for row in range(3):
        for col in range(3):
            if board_values[box_x + row][box_y + col] == test_num:
                return False
    return True


def solve_sudoku(sudoku_board):
    for row_idx in range(9):
        for col_idx in range(9):
            if sudoku_board[row_idx][col_idx] == 0:
                for test_value in range(1, 10):
                    if available_placements(row_idx, col_idx, test_value, sudoku_board):
                        sudoku_board[row_idx][col_idx] = test_value
                        if solve_sudoku(sudoku_board) is not None:
                            return sudoku_board
                        sudoku_board[row_idx][col_idx] = 0
                return
    print('Solution: \n', np.matrix(sudoku_board))
    return sudoku_board


def modified_array(nums_scanned, solution_nums):
    not_shared = []
    for idx, value in enumerate(solution_nums):
        if nums_scanned[idx] == solution_nums[idx]:
            not_shared.append(0)
        else:
            not_shared.append(value)
    return not_shared


def process_image(image, photo=False, debug=False):
    img_gray = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    print('Processing image....')
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    thresh = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    inverted = cv.bitwise_not(thresh)

    cont = cv.findContours(inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]
    img_cont = cv.drawContours(image.copy(), cont, -1, (0, 255, 0), 5)

    biggest, max_size = largest_contour(cont)

    if biggest.size != 0:
        fix_points = corrected_points(biggest)
        puzzle = cv.drawContours(image.copy(), biggest, -1, (0, 0, 255), 20)
        pts1 = np.float32(fix_points)
        pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)

        img_warp = cv.warpPerspective(image.copy(), matrix, (img_width, img_height))
        warp_bw = cv.cvtColor(img_warp, cv.COLOR_BGR2GRAY)
        warp_thresh = cv.adaptiveThreshold(warp_bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        warp_inverted = cv.bitwise_not(warp_thresh)

        boxes = finding_digits(warp_inverted)
        nums_found = []
        for box in boxes:
            if photo:
                box_prep = prep_box(box, photo=True)
            else:
                box_prep = prep_box(box)
            prediction = model.predict(box_prep)
            class_idx = np.argmax(prediction, axis=-1)
            probability = np.amax(prediction)

            if probability > 0.70:
                nums_found.append(class_idx[0])
            else:
                nums_found.append(0)

        display_num = img_blank.copy()
        img_detected = display_numbers(display_num, nums_found, color=(0, 0, 255))
        new_digits = img_blank.copy()

        cv.imshow('test', img_detected)
        cv.waitKey(0)
        cv.destroyAllWindows()

        board = np.array_split(nums_found, 9)
        print('Beginning to solve the puzzle....')
        solve_sudoku(board)

        solved_board = np.array(board).flatten()
        solved_digits = modified_array(nums_found, solved_board)
        digits_img = display_numbers(new_digits, solved_digits)

        paste_results = cv.addWeighted(digits_img, 1, img_warp.copy(), 0.5, 1)
        if debug:
            img_array = ([image, inverted, img_cont, puzzle], [img_warp, img_detected, digits_img, paste_results])
            img_stack = stack_images(0.5, img_array)
            cv.imshow('Sudoku Outline', img_stack)
            cv.waitKey(0)

        if photo:
            return paste_results
        else:
            return solved_board, paste_results


def fill_board(solution_values):
    print('\nFilling Board. Do not engage with keyboard!!!\n')
    titles = py.getAllTitles()
    desired_win = [win_name for win_name in titles if 'sudoku puzzles' in win_name][0]

    sudoku_win = py.getWindowsWithTitle(desired_win)[0]

    sudoku_win.activate()

    count = []
    for value in solution_values:
        py.press(str(value))
        py.press('right')
        count.append(0)

        if len(count) == 81:
            break
        elif len(count) % 9 == 0:
            py.press('down')
            py.press('left', presses=8)


def sudoku_screenshot():
    titles = py.getAllTitles()
    desired_win = [win_name for win_name in titles if 'sudoku puzzles' in win_name][0]

    sudoku_win = py.getWindowsWithTitle(desired_win)[0]

    sudoku_win.resizeTo(1294, 1000)
    sudoku_win.moveTo(1273, 0)
    sudoku_win.activate()
    time.sleep(2)
    py.click(x=1350, y=285)
    time.sleep(1)
    sudoku_img = py.screenshot(region=(1320, 255, 509, 508))
    cv_sudoku = cv.cvtColor(np.array(sudoku_img), cv.COLOR_RGB2BGR)

    return cv_sudoku


def play_again(chrome_driver=None, image=False):
    if image:
        while True:
            play = input('\nDo you want to solve another Sudoku? (y/n): ')
            if play.lower() == 'y':
                return True
            elif play.lower() == 'n':
                return False
            else:
                print('Not a valid input! Type in "y" for yes or "n" for no.')
    else:
        while True:
            play = input('Do you want to solve another Sudoku? (y/n): ')
            if play.lower() == 'y':
                return True
            elif play.lower() == 'n':
                chrome_driver.quit_chrome()
                return False
            else:
                print('Not a valid input! Type in "y" for yes or "n" for no.')


def image_type():
    img_form = input('Is the image a photo or a screenshot? (photo/screen): ')
    if img_form.lower() == 'photo':
        return True
    elif img_form.lower() == 'screen':
        return False
    else:
        print('Not a valid input. Type "photo" or "screen".')
        image_type()


def input_photo(photo_debug=False):
    img_format = image_type()

    file_loc = input('Input the path of the image: ')
    file_loc.replace('\\', '/')
    img_file = cv.imread(file_loc)

    if img_format:
        try:
            processed_input = process_image(img_file, photo=img_format, debug=photo_debug)
            return processed_input
        except AttributeError:
            print('Not a valid file path. Try again.')
            input_photo()
    else:
        try:
            processed_input = process_image(img_file, debug=photo_debug)[1]
            return processed_input
        except AttributeError:
            print('Not a valid file path. Try again.')
            input_photo()


def change_type(chrome_driver=None, image=False):
    user_cmd = input('Do you want to change the input type? (y/n): ')
    if image:
        if user_cmd.lower() == 'y':
            return True
        elif user_cmd.lower() == 'n':
            return False
        else:
            print('Not a valid input. Try again.')
            change_type()
    else:
        if user_cmd.lower() == 'y':
            chrome_driver.quit_chrome()
            return True
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
        url = 'https://sudoku.com/expert/'
        sud_web = StartDriver().desired_site(url)
        done = False
        while not done:
            sudoku_board = sudoku_screenshot()
            final_nums = process_image(sudoku_board)[0]
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
            display_results = input_photo()
            cv.imshow('Solution', display_results)
            cv.waitKey(0)
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
