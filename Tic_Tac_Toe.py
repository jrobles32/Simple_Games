import random


def display():
    """
    Prints a display showing user where numerical inputs will be placed. Returns a list of '-' representing potential
    placement locations.

    Returns:
        display_update (list): list of ten elements. Nine are potential placements for 'X' or 'O'.
    """
    display_fig = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    display_changes(display_fig)
    display_update = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return display_update


def display_changes(table):
    """
    Prints a display showing the current values in index one through nine of the input table.

    Parameters:
        table (list): a list of ten elements.

    Returns:
        none
    """
    print(table[7] + "|" + table[8] + "|" + table[9] +
          "\n" + table[4] + "|" + table[5] + "|" + table[6] +
          "\n" + table[1] + "|" + table[2] + "|" + table[3])


def input_logic(table):
    """
    Obtains an integer input between 1-9. If integer is represented by '-' in table, it is replaced with an 'X'.
    Returns an updated table list and prints a display of it.

    Parameters:
        table (list): a list of ten elements.

    Returns:
        display_changes(table): print of updated table list.
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

    Parameters:
        table (list): a list of ten elements.

    Returns:
        display_changes(table): print of updated table list.
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


def win_logic(table, win_print=False):
    """
    Determines if 'X' or 'O' appear three times in a row. Returns a boolean value based upon if a winner is found.

    Parameters:
        table (list): a list of ten elements.
        win_print (bool): 'True' to print who won, 'False' otherwise.

    Returns:
        bool: 'True' if a winner is found, 'False' otherwise.
    """
    # All possible winning combinations.
    if table[1] == table[2] == table[3] != '-':
        winner = table[1]

    elif table[4] == table[5] == table[6] != '-':
        winner = table[4]

    elif table[7] == table[8] == table[9] != '-':
        winner = table[7]

    elif table[1] == table[4] == table[7] != '-':
        winner = table[1]

    elif table[2] == table[5] == table[8] != '-':
        winner = table[2]

    elif table[3] == table[6] == table[9] != '-':
        winner = table[3]

    elif table[1] == table[5] == table[9] != '-':
        winner = table[1]

    elif table[3] == table[5] == table[7] != '-':
        winner = table[3]
    else:
        return False
    
    # Checking to see who won the game.
    if winner == 'X':
        if win_print:
            print('X is the Winner! Congratulations!')
        return True
    if winner == 'O':
        if win_print:
            print('O is the Winner! Better luck next time!')
        return True


def board_full(table):
    """
    Finds how many times '-' appears in table list to determine if the game is a draw. Returns a boolean value based on
    the frequency of '-'.

    Parameters:
        table (list): a list of ten elements.

    Returns:
        bool: 'True' if '-' appears only once in table, 'False' otherwise.
    """
    if table.count('-') > 1:
        return False
    
    else:
        print('The game is a draw!')
        return True


def play_again():
    """
    Determines if user wants to play another round. Returns a boolean value based on user input.

    Returns:
        bool: 'True' if user wants to play again, 'False' otherwise.
    """
    while True:
        user_input = input("Do you want to play again? (yes/no): ")
        if user_input.lower() == 'n' or user_input.lower() == 'no':
            print('Thanks for playing!')
            return False
        
        elif user_input.lower() == 'y' or user_input.lower() == 'yes':
            return True
        
        else:
            print('Not a valid input! Try Again!')
            continue


def main():
    while True:
        table = display()
        while True:
            input_logic(table)
            if win_logic(table, win_print=True):
                break

            if board_full(table):
                break

            ai_logic(table)
            if win_logic(table, win_print=True):
                break

        if not play_again():
            break


if __name__ == "__main__":
    main()
