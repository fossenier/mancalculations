from culations import MancalaBoard
from visuals import draw_footer_pit_selection, draw_header, draw_mancala_board
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
    draw_footer_pit_selection(stdscr, 9, 0)
#
#
#


stdscr.refresh()
stdscr.getch()
