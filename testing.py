from culations import *
import curses as curses

board = MancalaBoard()

stdscr = curses.initscr()
draw_store(stdscr, board, 0, 0, 0)
