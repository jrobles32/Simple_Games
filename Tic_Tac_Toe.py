def display():
    display_fig = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    display_changes(display_fig)
    display_update = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return display_update


def display_changes(table):
    print(table[7] + "|" + table[8] + "|" + table[9] +
          "\n" + table[4] + "|" + table[5] + "|" + table[6] +
          "\n" + table[1] + "|" + table[2] + "|" + table[3])


def input_logic(table):
    user_input = input("Where on the board do you want to place an X? Input an integer between 1 and 9: ")
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

    if table[user_input].isalpha():
        print('Number in use! Input another number!')
        return input_logic(table)
    else:
        table[user_input] = 'X'
        return display_changes(table)


def ai_logic(table):
    import random
    print("It is currently O's turn.")
    available_moves = []
    for move, index_value in enumerate(table):
        if index_value == '-' and move != 0:
            available_moves.append(move)

    for figure in ['O', 'X']:
        for value in available_moves:
            table_copy = table.copy()
            table_copy[value] = figure
            if win_logic(table_copy):
                table[value] = 'O'
                return display_changes(table)

    available_corners = []
    for values in available_moves:
        if values in [1, 3, 7, 9]:
            available_corners.append(values)
    if len(available_corners) > 0:
        random_corner = random.choice(available_corners)
        table[random_corner] = 'O'
        return display_changes(table)

    if table[5] in available_moves:
        table[5] = 'O'
        return display_changes(table)

    available_sides = []
    for values in available_moves:
        if values in [2, 4, 6, 8]:
            available_sides.append(values)
    if len(available_sides) > 0:
        random_side = random.choice(available_sides)
        table[random_side] = 'O'
        return display_changes(table)


def win_logic(table, win_print=False):
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

    if winner == 'X':
        if win_print:
            print('X is the Winner! Congratulations!')
        return True
    if winner == 'O':
        if win_print:
            print('O is the Winner! Better luck next time!')
        return True


def board_full(table):
    if table.count('-') > 1:
        return False
    else:
        print('The game is a draw!')
        return True


def game():
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


def main():
    import time
    done = False
    while not done:
        user_input = input("Do you want to play the game? (yes/no) ")
        if user_input.isalpha():
            if user_input.lower() == 'n' or user_input.lower() == 'no':
                print('Thanks for playing!')
                time.sleep(2)
                done = True
            elif user_input.lower() == 'y' or user_input.lower() == 'yes':
                game()
            else:
                print('Not a valid input! Try Again!')
                main()
        else:
            print('Not a valid input! Try Again!')
            main()


if __name__ == "__main__":
    main()
