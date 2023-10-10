"""
This is the backend code for my Mancala game.
It provides all the functions needed to run the game.
"""

import curses as C
import time as T

PIT_WIDTH = 3
PLAYER_PIT_COUNT = 6
STORE_WIDTH = 5


def move_rocks(board, player_turn, pit_selection):
    active_player = player_turn
    # Pick up rocks
    rock_count, board.p_pits[active_player][pit_selection] = (
        board.p_pits[active_player][pit_selection],
        0,
    )
    for _ in range(rock_count):
        pit_selection += 1
        # Add to store when at the end of the board, switch players, reset pit selection
        if pit_selection == PLAYER_PIT_COUNT:
            board.p_store[active_player] += 1
            active_player = 1 - active_player
            pit_selection = -1
        # Add to curret pit
        else:
            board.p_pits[active_player][pit_selection] += 1
    return board, active_player, pit_selection


def run_turn(stdscr, board, player_turn):
    """
    Runs one full user turn of mancalculations.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n

    Returns:
        `board`: Updated game state.
    """
    draw_board(stdscr, board, player_turn)
    print(
        f"Current board state:\nPlayer 2: {board.p_pits[1][::-1]}\nPlayer 2 Store: {board.p_store[1]}\nPlayer 1: {board.p_pits[0]}\nPlayer 1 Store: {board.p_store[0]}"
    )
    print(f"Taking player {player_turn + 1}'s turn")
    # Get player's choice of move
    pit_selection = get_player_move(player_turn)
    # Run player's move
    board, active_player, pit_selection = move_rocks(board, player_turn, pit_selection)
    # Give the player another turn if the last rock landed in the player's store
    if pit_selection == -1:
        print("Another turn!")
        board = run_turn(stdscr, board, player_turn)
    # Check if a steal is possible
    else:
        board = steal_rocks(board, active_player, pit_selection)
    return board


def draw_board(stdscr, board, player_turn):
    """
    Illustrates the game board in the terminal.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
    """
    # Reset screen, draw title, draw player turn
    stdscr.clear()
    stdscr.addstr(0, 0, "Mancalculations")
    stdscr.addstr(1, 0, f"Player {board.player_turn + 1}'s turn")

    draw_store(stdscr, board, player_turn, 3, 0)

    return


def draw_mancala_board(stdscr, board, vertical_offfset, horizontal_offset):
    draw_store(stdscr, board, 0, vertical_offfset, horizontal_offset)
    horizontal_offset += 3
    for i in range(2):
        for j in range(PLAYER_PIT_COUNT):
            draw_pit(
                stdscr,
                board,
                i,
                vertical_offfset + i * 2,
                horizontal_offset + (j + 1) * 4,
                j,
            )
    draw_store(
        stdscr, board, 1, vertical_offfset, horizontal_offset + PLAYER_PIT_COUNT * 4 - 1
    )


def draw_store(stdscr, board, player_turn, vertical_offfset, horizontal_offset):
    """
    Draws a player Mancala store at the given offset location.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
    """
    lines_to_draw = [
        "-" * (STORE_WIDTH + 2),
        f"|{'':^{STORE_WIDTH}}|",
        f"|{board.p_store[player_turn]:^{STORE_WIDTH}}|",
        f"|{'':^{STORE_WIDTH}}|",
        "-" * (STORE_WIDTH + 2),
    ]

    for line in lines_to_draw:
        stdscr.addstr(vertical_offfset, horizontal_offset, line)
        vertical_offfset += 1


def draw_pit(
    stdscr, board, player_turn, vertical_offset, horizontal_offset, pit_selection
):
    """
    Draws a player Mancala pit at the given offset location.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
        `pit_selection` (`integer`): Chosen player pit.\n"""
    lines_to_draw = [
        "-" * (PIT_WIDTH + 1),
        f"{board.p_pits[player_turn][pit_selection]:^{PIT_WIDTH}}|",
        "-" * (PIT_WIDTH + 1),
    ]

    for line in lines_to_draw:
        stdscr.addstr(vertical_offset, horizontal_offset, line)
        vertical_offset += 1


def steal_rocks(board, active_player, pit_selection):
    # Check if the last rock landed in an empty pit on the player's side
    if (
        board.p_pits[active_player][pit_selection] == 1
        and board.p_pits[1 - active_player][(PLAYER_PIT_COUNT - 1) - pit_selection] != 0
    ):
        print("Steal!")
        # Steal the rocks from the opposite pit
        board.p_store[active_player] += (
            board.p_pits[active_player][pit_selection]
            + board.p_pits[1 - active_player][(PLAYER_PIT_COUNT - 1) - pit_selection]
        )
        board.p_pits[active_player][pit_selection] = 0
        board.p_pits[1 - active_player][(PLAYER_PIT_COUNT - 1) - pit_selection] = 0
    return board


def is_game_over(board):
    # Check if the game is over
    for player in range(2):
        if sum(board.p_pits[player]) == 0:
            return True
    return False


def score(board):
    # Add up the score
    p1_score = board.p_store[0] + sum(board.p_pits[0])
    p2_score = board.p_store[1] + sum(board.p_pits[1])
    return p1_score, p2_score


def run_game():
    stdscr = C.initscr()
    game_is_over = False
    player_turn = 0
    board = MancalaBoard()
    while not game_is_over:
        board = run_turn(stdscr, board, player_turn)
        game_is_over = is_game_over(board)
        player_turn = 1 - player_turn
    return score(board)


def get_player_move(player_turn):
    print("Getting player move")
    pit_selection = -1
    # Check if the pit is a valid choice from 0-5
    while pit_selection < 0 or pit_selection > PLAYER_PIT_COUNT - 1:
        try:
            # The player sees pits 1-6, but the code uses 0-5
            pit_selection = int(input("Choose a valid pit (1-6): ")) - 1
        except TypeError:
            print("Sorry, that was an invalid input. Please enter 1, 2, 3, 4, 5, or 6.")
    if player_turn == 1:
        # The player sees pits 1-6, but the code uses 0-5
        pit_selection = (PLAYER_PIT_COUNT - 1) - pit_selection
    return pit_selection


class MancalaBoard:
    def __init__(self):
        self.p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        self.p_store = [0, 0]
