import cv2 as cv
from pathlib import Path
import numpy as np


def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[1] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


userValue = int(input("what number do you want: "))

rows = 0
while rows <= userValue - 1:
    leftColumn = 1
    while leftColumn <= userValue:
        if leftColumn >= userValue - rows:
            print('*', end=' ')
        else:
            print(' ', end=' ')
        leftColumn += 1

    rightColumn = userValue - 1
    while rightColumn >= 1:
        if rightColumn >= userValue - rows:
            print('-', end=' ')
        else:
            print(' ', end=' ')
        rightColumn -= 1

    print()
    rows += 1

rows = 0
step = 1
while rows <= userValue - 1:
    leftColumn = 1
    while leftColumn <= userValue:
        if leftColumn > step:
            print('-', end=' ')
        else:
            print(' ', end=' ')
        leftColumn += 1
    step += 1

    rightColumn = userValue - 1
    while rightColumn >= 1:
        if rightColumn >= step:
            print('*', end=' ')
        else:
            print(' ', end=' ')
        rightColumn -= 1

    print()
    rows += 1


triangleRows = 0
while triangleRows <= userValue - 1:
    triangleColumns = 1
    while triangleColumns <= userValue:
        if triangleColumns >= userValue - triangleRows:
            print('*', ' ', end=' ')
        else:
            (print(' ', end=' '))
        triangleColumns += 1

    print()
    triangleRows += 1

userInput = int(input("Input an integer: "))

for row in range(userInput):
    for leftCol in range(1, userInput+1):
        if leftCol >= userInput - row:
            print(leftCol, end=" ")
        else:
            print(' ', end=' ')

    for rightCol in reversed(range(1, userInput)):
        if rightCol >= userInput - row:
            print(rightCol, end=" ")

    print()

for row in range(userInput):
    for leftCol in range(1, userInput+1):
        if leftCol > row + 1:
            print(leftCol, end=" ")
        else:
            print(' ', end=' ')

    for rightCol in reversed(range(1, userInput)):
        if rightCol > row + 1:
            print(rightCol, end=" ")

    print()

for row in range(userInput + 1):
    for col in range(userInput):
        if col >= userInput - row:
            print('*', ' ', end=' ')
        else:
            print(' ', end=' ')
    print()
