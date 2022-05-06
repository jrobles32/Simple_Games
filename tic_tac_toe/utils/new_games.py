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
