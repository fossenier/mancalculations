PIT_COUNT = 6


def main():
    print("Mancala is starting")
    p1_score, p2_score = run_game()
    return


def move_pit(board, player_turn, pit_selection):
    # Implement the game logic here to update the board state after a move
    rock_count, board.p_pits[player_turn][pit_selection] = board.p_pits[player_turn][pit_selection], 0
    for i in range(rock_count):
        active_pit = (pit_selection + i + 1)
    pass


def is_game_over(board):
    # Check if the game is over according to the rules
    pass


def score(self):
    # Determine the winner or declare a tie
    pass


def run_game():
    game_is_over = False
    player_turn = 0
    board = MancalaBoard()
    print(board.p1_pits, board.p1_store, board.p2_pits, board.p2_store)
    while not game_is_over:
        pit_selection = select_pit(board, player_turn)
        move_pit(board, player_turn, pit_selection)
        game_is_over = is_game_over(board)
        player_turn = 1 - player_turn
    return score(board)


def select_pit():
    pit_selection = -1
    while pit_selection < 0 or pit_selection > PIT_COUNT:
        try:
            pit_selection = int(input("Enter a valid pit (0-6): "))
        except TypeError:
            print("Sorry, that was an invalid input, please enter an integer [0, 6]")
    return pit_selection


class MancalaBoard:
    def __init__(self):
        self.p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        self.p_store = [0, 0]


if __name__ == "__main__":
    main()
