"""
This is the frontend code for my Mancala game.
It runs the game and provides the user interface.
"""

from constants import (
    VERTICAL_OFFSET_OF_ERROR,
    VERTICAL_OFFSET_OF_PROMPT,
    MESSAGE_WAIT,
)
from culations import check_game_over, MancalaBoard, move_rocks, score_game, steal_rocks
from visuals import (
    draw_blank,
    draw_header,
    draw_mancala_board,
    draw_pit_selection,
    draw_message,
)
import curses as curses
import time as time


def main(stdscr):
    """
    Runs a game of Manncala for the user.

    Args:
        `stdscr` (`stdscr`): Cureses main window.
    """
    # clear screen and hide cursor
    stdscr.clear()
    curses.curs_set(0)

    # setup initial game state
    board = MancalaBoard()
    is_game_over = False
    player_turn = 1
    draw_mancala_board(stdscr, board, 3, 0)

    # main game loop
    while not is_game_over:
        player_turn = 1 - player_turn
        draw_header(stdscr, player_turn)
        run_turn(stdscr, board, player_turn)
        is_game_over = check_game_over(board)

    # display game over message
    p1_score, p2_score = score_game(board)


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
        draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
        curses.curs_set(1)
        # get player input
        input = draw_pit_selection(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0)
        try:
            pit_selection = int(input)
            # display error if number is out of range
            if pit_selection < 1 or 6 < pit_selection:
                draw_blank(stdscr, VERTICAL_OFFSET_OF_ERROR, 0, 1)
                draw_message(
                    stdscr,
                    VERTICAL_OFFSET_OF_ERROR,
                    0,
                    "Invalid number. Must enter 1, 2, 3, 4, 5, or 6.",
                )
        # display error if input is not a number
        except ValueError:
            draw_blank(stdscr, VERTICAL_OFFSET_OF_ERROR, 0, 1)
            draw_message(
                stdscr,
                VERTICAL_OFFSET_OF_ERROR,
                0,
                "Invalid input. Please enter a number.",
            )
    # hide cursor and clear prompt and potential error message
    curses.curs_set(0)
    draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 2)
    return pit_selection - 1


def run_turn(stdscr, board, player_turn):
    """
    Runs one full user turn of mancalculations.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.
    """
    is_steal = False
    pit_selection = get_player_move(stdscr)
    active_side, active_pit = move_rocks(board, player_turn, pit_selection)
    draw_mancala_board(stdscr, board, 3, 0)
    # give the player another turn if they landed in their store
    if active_pit == -1:
        draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
        draw_message(
            stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, "Nice one, take another turn!"
        )
        time.sleep(MESSAGE_WAIT)
        draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
        run_turn(stdscr, board, player_turn)
    # if the active side is the player's, check if a steal occurs
    else:
        is_steal = steal_rocks(board, active_side, player_turn, active_pit)
        if is_steal:
            draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
            draw_message(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, "Steal!")
            time.sleep(MESSAGE_WAIT)
            draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
    if not is_steal:
        draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)
        draw_message(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, "Next player's turn.")
        time.sleep(MESSAGE_WAIT)
        draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, 0, 1)


if __name__ == "__main__":
    curses.wrapper(main)
