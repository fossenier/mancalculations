"""
Handles running a game of Mancala visually.
"""

from mancala import Mancala, TURN_HUMAN, TURN_AI


def main():
    """
    Runs a game of Mancala for the user.
    """
    # setup initial game state
    board = Mancala()
    is_game_over = False
    player_turn = board.p_turn()

    # main game loop
    while not is_game_over:
        print(board)
        if player_turn == TURN_AI:
            move = board.minimax()
            board = board.result(move)
            print(f"AI plays {move}")
        else:
            while True:
                try:
                    move = int(input("Enter your move: "))
                    # Assume 'valid_move' is a function that checks if the move is allowed on the board
                    board = board.result(move)
                    break  # Exit the loop if the move is valid and successfully made

                except ValueError:
                    print("Please enter a valid integer.")

        player_turn = board.p_turn()
        is_game_over = board.terminal()

    # display game over message
    print(board.utility())
    print("Game over.")


if __name__ == "__main__":
    main()
