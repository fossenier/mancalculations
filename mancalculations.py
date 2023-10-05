PIT_COUNT = 6


def main():
    run_game()
    # The game will be broken into Player 0 and Player 1 turns
    # p_turn = 0
    # game_winner = None
    # try:
    #     while True:
    #         pit_choice = input("Select a pit to move: ")
    #         game_winner = board.make_move(p_turn, pit_choice)
    #         p_turn = 1 - p_turn
    #         if game_winner is not None:
    #             break
    #         pass
    # except KeyboardInterrupt:
    #     print("Exiting game...")
    return


def make_move(self, player, pit_index):
    # Implement the game logic here to update the board state after a move
    pass


def is_game_over(self):
    # Check if the game is over according to the rules
    pass


def score(self):
    # Determine the winner or declare a tie
    pass


def run_game():
    game_is_over = False
    player_turn = 0
    board = MancalaBoard()
    print("Mancala is starting")
    print(board.p1_pits, board.p1_store, board.p2_pits, board.p2_store)
    while not game_is_over:
        pit_selection = take_turn(board, player_turn)
        player_turn = 1 - player_turn
    return


def take_turn(board, player_turn):
    print("aaaa")
    pit_selection = -1
    while pit_selection < 0 or pit_selection > PIT_COUNT:
        try:
            pit_selection = int(input("input: "))
        except TypeError:
            print("Sorry, that was an invalid input, please enter an integer [0, 6]")
    return pit_selection


class MancalaBoard:
    def __init__(self):
        self.p1_pits = [4, 4, 4, 4, 4, 4]
        self.p2_pits = [4, 4, 4, 4, 4, 4]
        self.p1_store = 0
        self.p2_store = 0


if __name__ == "__main__":
    main()
