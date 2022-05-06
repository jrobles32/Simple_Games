def win_logic(table, win_print=False):
    """
    Determines if 'X' or 'O' appear three times in a row. Returns a boolean value based upon if a winner is found.

    :param table: A list of ten elements.
    :type table: list
    :param win_print: 'True' to print who won, 'False' otherwise.
    :type win_print: bool
    :returns: 'True' if a winner is found, 'False' otherwise.
    :rtype: bool
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

    :param table: A list of ten elements.
    :type table: list
    :returns: 'True' if '-' appears only once in table, 'False' otherwise.
    :rtype: bool
    """
    if table.count('-') > 1:
        return False
    
    else:
        print('The game is a draw!')
        return True
