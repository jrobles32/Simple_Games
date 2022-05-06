from display.terminal_outputs import display
from game_logic.placement_controls import input_logic, ai_logic
from game_logic.board_status import win_logic, board_full
from utils.new_games import play_again

def main():
    outer_cont = True
    while outer_cont:
        inner_cont = True
        table = display()

        while inner_cont:
            input_logic(table)
            if win_logic(table, win_print=True):
                inner_cont = False

            if board_full(table):
                inner_cont = False

            ai_logic(table)
            if win_logic(table, win_print=True):
                inner_cont = False

        if not play_again():
            outer_cont = False


if __name__ == "__main__":
    main()
