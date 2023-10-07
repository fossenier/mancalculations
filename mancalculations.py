PIT_COUNT = 6


def main():
    print("Mancala is starting")
    p1_score, p2_score = run_game()
    return


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
        if pit_selection == PIT_COUNT:
            board.p_store[active_player] += 1
            active_player = 1 - active_player
            pit_selection = -1
        # Add to curret pit
        else:
            board.p_pits[active_player][pit_selection] += 1
    return board, active_player, pit_selection


def take_turn(board, player_turn):
    print("\n\n\n")
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
        board = take_turn(board, player_turn)
    # Check if a steal is possible
    else:
        board = steal_rocks(board, active_player, pit_selection)
    return board


def steal_rocks(board, active_player, pit_selection):
    # Check if the last rock landed in an empty pit on the player's side
    if (
        board.p_pits[active_player][pit_selection] == 1
        and board.p_pits[1 - active_player][(PIT_COUNT - 1) - pit_selection] != 0
    ):
        print("Steal!")
        # Steal the rocks from the opposite pit
        board.p_store[active_player] += (
            board.p_pits[active_player][pit_selection]
            + board.p_pits[1 - active_player][(PIT_COUNT - 1) - pit_selection]
        )
        board.p_pits[active_player][pit_selection] = 0
        board.p_pits[1 - active_player][(PIT_COUNT - 1) - pit_selection] = 0
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
    game_is_over = False
    player_turn = 0
    board = MancalaBoard()
    while not game_is_over:
        board = take_turn(board, player_turn)
        game_is_over = is_game_over(board)
        player_turn = 1 - player_turn
    return score(board)


def get_player_move(player_turn):
    print("Getting player move")
    pit_selection = -1
    # Check if the pit is a valid choice from 0-5
    while pit_selection < 0 or pit_selection > PIT_COUNT - 1:
        try:
            # The player sees pits 1-6, but the code uses 0-5
            pit_selection = int(input("Choose a valid pit (1-6): ")) - 1
        except TypeError:
            print("Sorry, that was an invalid input. Please enter 1, 2, 3, 4, 5, or 6.")
    if player_turn == 1:
        # The player sees pits 1-6, but the code uses 0-5
        pit_selection = (PIT_COUNT - 1) - pit_selection
    return pit_selection


class MancalaBoard:
    def __init__(self):
        self.p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        self.p_store = [0, 0]


if __name__ == "__main__":
    main()
