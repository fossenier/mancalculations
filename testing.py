from culations import MancalaBoard
from visuals import (
    draw_blank,
    draw_header,
    draw_mancala_board,
    draw_pit_selection,
    draw_error_message,
)
import curses as curses
import time as T

board = MancalaBoard()

stdscr = curses.initscr()
# Clear screen, hide the cursor
stdscr.clear()
curses.curs_set(0)

# Run a full turn
#
#
#
desc = """
Runs one full user turn of mancalculations.

Args:
    `stdscr` (`stdscr`): Curses main window.\n
    `board` (`MancalaBoard`): Current game state.\n
    `player_turn` (`integer`): Current player turn.\n

Returns:
    `board`: Updated game state.
"""
draw_header(stdscr, 0)
draw_mancala_board(stdscr, board, 3, 0)
pit_selection = -1
while pit_selection < 1 or 6 < pit_selection:
    curses.curs_set(1)
    draw_blank(stdscr, 9, 0, 1)
    input = draw_pit_selection(stdscr, 9, 0)
    try:
        pit_selection = int(input)
        if pit_selection < 1 or 6 < pit_selection:
            draw_blank(stdscr, 10, 0, 1)
            draw_error_message(
                stdscr, 10, 0, "Invalid number. Must enter 1, 2, 3, 4, 5, or 6."
            )
    except ValueError:
        draw_blank(stdscr, 10, 0, 1)
        draw_error_message(stdscr, 10, 0, "Invalid input. Please enter a number.")
curses.curs_set(0)
draw_blank(stdscr, 9, 0, 2)
#
#
#


stdscr.refresh()
stdscr.getch()
