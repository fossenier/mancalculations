"""
This is the user facing code for my Mancala game.
It runs the game and provides the user interface.
"""


import curses as curses
import time as time

from constants import (
    HORIZONTAL_OFFSET,
    VERTICAL_OFFSET_OF_BOARD,
    VERTICAL_OFFSET_OF_ERROR,
    VERTICAL_OFFSET_OF_PROMPT,
    MESSAGE_WAIT,
    VALID_USER_PITS,
)
from culations import Culations
from visuals import Visuals


class MancalaBoard(Culations, Visuals):
    def __init__(self):
        # basic game state
        self.p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        # TODO remove, for testing only
        # self.p_pits = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
        self.p_store = [0, 0]


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
    board.draw_mancala_board(stdscr, VERTICAL_OFFSET_OF_BOARD, HORIZONTAL_OFFSET)

    # main game loop
    while not is_game_over:
        player_turn = 1 - player_turn
        board.draw_header(stdscr, player_turn)
        run_turn(stdscr, board, player_turn)
        is_game_over = board.check_game_over()

    # display game over message
    p1_score, p2_score = board.score_game()
    board.draw_game_over_animation(stdscr, p1_score, p2_score)


def get_player_move(board, stdscr):
    """
    Gets the player's choice of move.

    Args:
        `stdscr` (`stdscr`): Curses main window.

    Returns:
        `pit_selection` (`integer`): Player's selected pit of choice.
    """
    # initialize pit selection
    pit_selection = -1
    while pit_selection not in VALID_USER_PITS:
        # show cursor and clear previous prompt
        board.draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, HORIZONTAL_OFFSET, 1)
        curses.curs_set(1)
        # get player input
        input = board.draw_pit_selection(
            stdscr, VERTICAL_OFFSET_OF_PROMPT, HORIZONTAL_OFFSET
        )
        try:
            pit_selection = int(input)
            # display error if number is out of range
            if pit_selection not in VALID_USER_PITS:
                board.draw_blank(stdscr, VERTICAL_OFFSET_OF_ERROR, HORIZONTAL_OFFSET, 1)
                board.draw_text(
                    stdscr,
                    VERTICAL_OFFSET_OF_ERROR,
                    0,
                    "Invalid number. Must enter 1, 2, 3, 4, 5, or 6.",
                )
        # display error if input is not a number
        except ValueError:
            board.draw_blank(stdscr, VERTICAL_OFFSET_OF_ERROR, HORIZONTAL_OFFSET, 1)
            board.draw_text(
                stdscr,
                VERTICAL_OFFSET_OF_ERROR,
                0,
                "Invalid input. Please enter a number.",
            )
    # hide cursor and clear prompt and potential error message
    curses.curs_set(0)
    board.draw_blank(stdscr, VERTICAL_OFFSET_OF_PROMPT, HORIZONTAL_OFFSET, 2)
    return pit_selection - 1


def run_turn(stdscr, board, player_turn):
    """
    Runs one full user move in mancalculations.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.
    """
    is_steal = False
    valid_move = False
    while not valid_move:
        pit_selection = get_player_move(board, stdscr)
        if board.p_pits[player_turn][pit_selection] != 0:
            valid_move = True
        else:
            board.draw_message(
                stdscr,
                VERTICAL_OFFSET_OF_PROMPT,
                HORIZONTAL_OFFSET,
                "Invalid move. Please select a pit with rocks in it.",
                MESSAGE_WAIT,
            )
    active_side, active_pit = board.move_rocks(player_turn, pit_selection)
    board.draw_mancala_board(stdscr, VERTICAL_OFFSET_OF_BOARD, HORIZONTAL_OFFSET)
    # give the player another turn if they landed in their store
    if active_pit == -1:
        board.draw_message(
            stdscr,
            VERTICAL_OFFSET_OF_PROMPT,
            HORIZONTAL_OFFSET,
            "Nice one, take another turn!",
            MESSAGE_WAIT,
        )
        run_turn(stdscr, board, player_turn)
    # if the active side is the player's, check if a steal occurs
    else:
        is_steal = board.steal_rocks(active_side, player_turn, active_pit)
        if is_steal:
            board.draw_mancala_board(
                stdscr, VERTICAL_OFFSET_OF_BOARD, HORIZONTAL_OFFSET
            )
            board.draw_message(
                stdscr,
                VERTICAL_OFFSET_OF_PROMPT,
                HORIZONTAL_OFFSET,
                "Steal!",
                MESSAGE_WAIT,
            )
    if not is_steal:
        board.draw_message(
            stdscr,
            VERTICAL_OFFSET_OF_PROMPT,
            HORIZONTAL_OFFSET,
            "Next player's turn.",
            MESSAGE_WAIT,
        )


if __name__ == "__main__":
    curses.wrapper(main)
