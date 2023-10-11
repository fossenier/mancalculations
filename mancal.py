"""
This is the frontend code for my Mancala game.
It runs the game and provides the user interface.
"""

from culations import get_game_over, MancalaBoard, move_rocks, steal_rocks
import curses as curses
from visuals import (
    draw_blank,
    draw_header,
    draw_mancala_board,
    draw_pit_selection,
    draw_message,
)


def main(stdscr):
    # clear screen and hide cursor
    stdscr.clear()
    curses.curs_set(0)

    # setup initial game state
    board = MancalaBoard()
    is_game_over = False
    player_turn = 1

    # setup initial visual state
    draw_header(stdscr, 0)
    draw_mancala_board(stdscr, board, 3, 0)

    # main game loop
    while not is_game_over:
        player_turn = 1 - player_turn
        run_turn(stdscr, board, player_turn)
        is_game_over = get_game_over(board)


def get_player_move(stdscr):
    """
    Gets the player's choice of move.

    Args:
        `stdscr` (`stdscr`): Curses main window.

    Returns:
        `pit_selection` (`integer`): Player's selected pit of choice.
    """
    # initialize pit selection
    pit_selection = -1
    while pit_selection < 1 or 6 < pit_selection:
        # show cursor and clear previous prompt
        draw_blank(stdscr, 9, 0, 1)
        curses.curs_set(1)
        # get player input
        input = draw_pit_selection(stdscr, 9, 0)
        try:
            pit_selection = int(input)
            # display error if number is out of range
            if pit_selection < 1 or 6 < pit_selection:
                draw_blank(stdscr, 10, 0, 1)
                draw_message(
                    stdscr, 10, 0, "Invalid number. Must enter 1, 2, 3, 4, 5, or 6."
                )
        # display error if input is not a number
        except ValueError:
            draw_blank(stdscr, 10, 0, 1)
            draw_message(stdscr, 10, 0, "Invalid input. Please enter a number.")
    # hide cursor and clear prompt and potential error message
    curses.curs_set(0)
    draw_blank(stdscr, 9, 0, 2)
    return pit_selection - 1


def run_turn(stdscr, board, player_turn):
    pit_selection = get_player_move(stdscr)
    board, active_side, active_pit = move_rocks(board, player_turn, pit_selection)
    draw_mancala_board(stdscr, board, 3, 0)
    # give the player another turn if they landed in their store
    if active_pit == -1:
        draw_message(stdscr, 11, 0, "Nice one, take another turn!")
        board = run_turn(stdscr, board, player_turn)
    # if the active side is the player's, check if a steal occurs
    else:
        board, is_steal = steal_rocks(board, active_side, player_turn, active_pit)
        if is_steal:
            draw_message(stdscr, 11, 0, "Steal!")
    return board


if __name__ == "__main__":
    curses.wrapper(main)
