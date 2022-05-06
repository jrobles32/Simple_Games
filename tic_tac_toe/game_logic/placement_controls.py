import random

from display.terminal_outputs import display_changes
from game_logic.board_status import win_logic


def input_logic(table):
    """
    Obtains an integer input between 1-9. If integer is represented by '-' in table, it is replaced with an 'X'.
    Returns an updated table list and prints a display of it.

    :param table: A list of ten elements.
    :type table: list
    :returns: A print of updated table values.
    :rtype: list
    """
    user_input = input("Where on the board do you want to place an X? Input an integer between 1 and 9: ")
    
    # Validating user input.
    if user_input.isdigit():
        if 0 < int(user_input) < 10:
            user_input = int(user_input)
        else:
            print("You did not input an integer between 1 and 9! Try again!")
            return input_logic(table)
    elif user_input.isalpha():
        print("Please input an integer not a string!")
        return input_logic(table)
    else:
        print("Please input an integer! Your input was not valid!")
        return input_logic(table)
    
    # Checking if user input is in use and placing it in appropriate location.
    if table[user_input].isalpha():
        print('Number in use! Input another number!')
        return input_logic(table)
    else:
        table[user_input] = 'X'
        return display_changes(table)


def ai_logic(table):
    """
    Finds the best possible location to replace '-' in table list with an 'O'. Returns an updated table list and prints a
    display of it.

    :param table: A list of ten elements.
    :type table: list
    :returns: A pring of upated table values.
    :rtype: list
    """
    print("It is currently O's turn.")
    
    # Finding all empty spaces in list.
    available_moves = []
    for move, index_value in enumerate(table):
        if index_value == '-' and move != 0:
            available_moves.append(move)

    # Determining if its possible to block winning move for 'X' or to win game by placing 'O'.
    for figure in ['O', 'X']:
        for value in available_moves:
            table_copy = table.copy()
            table_copy[value] = figure

            if win_logic(table_copy):
                table[value] = 'O'
                return display_changes(table)

    # Finding available corners and placing 'O' randomly in one of them if they are empty.
    available_corners = []
    for values in available_moves:
        if values in [1, 3, 7, 9]:
            available_corners.append(values)
            
    if len(available_corners) > 0:
        random_corner = random.choice(available_corners)
        table[random_corner] = 'O'
        return display_changes(table)

    # Placing 'O' in center if it is empty.
    if table[5] in available_moves:
        table[5] = 'O'
        return display_changes(table)

    # Placing 'O' in random empty middle outer row/column position.
    available_sides = []
    for values in available_moves:
        if values in [2, 4, 6, 8]:
            available_sides.append(values)

    if len(available_sides) > 0:
        random_side = random.choice(available_sides)
        table[random_side] = 'O'
        return display_changes(table)
