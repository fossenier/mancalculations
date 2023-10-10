from culations import *
import curses as curses
import time as T

board = MancalaBoard()

stdscr = curses.initscr()
# Clear screen, hide the cursor
stdscr.clear()
curses.curs_set(0)

draw_pits(stdscr, board, 0, 0, 6, 0)
draw_pits(stdscr, board, 0, 0, 10, 0)
draw_pits(stdscr, board, 0, 0, 14, 0)

draw_pits(stdscr, board, 0, 2, 6, 0)
draw_pits(stdscr, board, 0, 2, 10, 0)
draw_pits(stdscr, board, 0, 2, 14, 0)


draw_store(stdscr, board, 0, 0, 0)

for i in range(2):
    for j in range (PLAYER_PIT_COUNT):
        draw_pits(stdscr, board, i, i * 2, )

wait = stdscr.getch()

for i in range(2):
    print(i)
