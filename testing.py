import curses
from visuals import draw_pit_selection

if __name__ == "__main__":
    curses.wrapper(draw_pit_selection, 9, 0)
