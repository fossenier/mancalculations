"""
This is backend code for my Mancala game.
It provides all the functions needed to run the game.
"""

from constants import PLAYER_PIT_COUNT


def move_rocks(board, player_turn, pit_selection):
    """
    Moves the rocks from the selected pit.

    Args:
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
        `pit_selection` (`integer`): Player's selected pit of choice.\n

    Returns:
        `board` (`MancalaBoard`): Updated game state.\n
        `player_turn` (`integer`): Most recent active player's side of the board.\n
        `pit_selection` (`integer`): Most recent active pit.\n
    """
    # track active side of the board, pick up rocks from selected pit
    active_player = player_turn
    rock_count, board.p_pits[active_player][pit_selection] = (
        board.p_pits[active_player][pit_selection],
        0,
    )
    # move rocks from selected pit
    while rock_count > 0:
        pit_selection += 1
        if pit_selection == PLAYER_PIT_COUNT:
            # add to store if the active side matches the player's turn
            if active_player == player_turn:
                board.p_store[active_player] += 1
                rock_count -= 1
            # switch active side of the board, reset pit selection
            active_player = 1 - active_player
            pit_selection = -1
        else:
            board.p_pits[active_player][pit_selection] += 1
            rock_count -= 1
    return board, active_player, pit_selection


def run_turn(stdscr, board, active_player):
    """
    Runs one full user turn of mancalculations.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n

    Returns:
        `board`: Updated game state.
    """
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


def steal_rocks(board, active_side, player_turn, active_pit):
    """
    Performs a steal if the last rock landed in an empty pit on the player's side.

    Args:
        `board` (`MancalaBoard`): Current game state.\n
        `active_side` (`integer`): Most recent active side of the board.\n
        `player_turn` (`integer`): Current player turn.\n
        `active_pit` (`integer`): Most recent active pit.\n

    Returns:
        `board` (`MancalaBoard`): Updated game state.\n
        `is_steal` (`boolean`): Whether or not a steal occurred.\n
    """
    if active_side != player_turn:
        return board, False
    is_steal = False
    # check if the last rock landed in an empty pit on the player's side
    if (
        board.p_pits[active_side][active_pit] == 1
        and board.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit] != 0
    ):
        is_steal = True
        # steal the rocks from the opposite pit
        board.p_store[active_side] += (
            board.p_pits[active_side][active_pit]
            + board.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit]
        )
        board.p_pits[active_side][active_pit] = 0
        board.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit] = 0
    return board, is_steal


def get_game_over(board):
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
    game_is_over = False
    player_turn = 0
    board = MancalaBoard()
    while not game_is_over:
        # board = run_turn(stdscr, board, player_turn)
        game_is_over = get_game_over(board)
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
