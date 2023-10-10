"""
This is the frontend code for my Mancala game.
It runs the game and provides the user interface.
"""

from culations import *
import curses


def main(stdscr):
    # Clear screen, hide cursor
    stdscr.clear()
    curses.curs_set(0)

    # Initialize game state
    board = MancalaBoard()
    game_is_won = False
    player_turn = 0

    # while not game_is_won:
    #     board = run_turn(board, player_turn)
    #     game_is_won = is_game_over(board)
    #     player_turn = 1 - player_turn
    # # return score(board)

    # Run game
    while not game_is_won:
        board = run_turn(stdscr, board, player_turn)

        stdscr.clear()
        draw_board(stdscr, board)
        stdscr.refresh()
        c = stdscr.getch()

        stdscr.addstr(10, 0, f"You pressed {c}!")
        stdscr.refresh()
        # Your game logic here


def draw_board(stdscr, board):
    # Line 0 game title
    stdscr.addstr(0, 0, "Mancalculations")

    # Line 1 player turn
    stdscr.addstr(1, 0, f"Player {board.player_turn + 1}'s turn")
    return


def draw_boards(stdscr, board):
    stdscr.addstr(0, 0, "Mancalculations")

    # Draw Player 2's pits
    stdscr.addstr(2, 0, "-" * 37)
    stdscr.addstr(3, 0, "|")

    for pit in board.p_pits[1][::-1]:
        stdscr.addstr(f"{pit:^5}|")

    stdscr.addstr(4, 0, "-" * 37)

    # Draw Player 1's store (on the left)
    stdscr.addstr(2, 0, "-----------")
    stdscr.addstr(3, 0, "|         |")
    stdscr.addstr(4, 0, f"|   {board.p_store[0]:^3}   |")
    stdscr.addstr(5, 0, "|         |")
    stdscr.addstr(6, 0, "-----------")

    # Draw Player 2's store (on the right)
    stdscr.addstr(2, 38, "-----------")
    stdscr.addstr(3, 38, "|         |")
    stdscr.addstr(4, 38, f"|   {board.p_store[1]:^3}   |")
    stdscr.addstr(5, 38, "|         |")
    stdscr.addstr(6, 38, "-----------")

    # Draw Player 1's pits
    stdscr.addstr(6, 0, "-" * 37)
    stdscr.addstr(7, 0, "|")

    for pit in board.p_pits[0]:
        stdscr.addstr(f"{pit:^5}|")

    stdscr.addstr(8, 0, "-" * 37)


if __name__ == "__main__":
    curses.wrapper(main)
