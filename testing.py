from culations import *
import curses as curses
import time as T

board = MancalaBoard()

stdscr = curses.initscr()
# Clear screen, hide the cursor
stdscr.clear()
curses.curs_set(0)

draw_mancala_board(stdscr, board, 0, 0)

stdscr.refresh()
stdscr.getch()
