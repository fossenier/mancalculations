"""
Handles running a game of Connect4 visually.
"""

from connect4 import Connect4, TURN_HUMAN, TURN_AI

def main():
    board = Connect4()
    is_game_over = False
    player_turn = board.p_turn()

    while not is_game_over:
        print(board)
        if player_turn == TURN_AI:
            move = board.minimax()
            board = board.result(move)
            print(f"AI plays column {move + 1}")
        else:
            while True:
                try:
                    move = int(input("Enter your move (1-7): ")) - 1
                    if move not in board.actions():
                        raise ValueError("Invalid move.")
                    board = board.result(move)
                    break
                except ValueError:
                    print("Please enter a valid move (1-7).")

        player_turn = board.p_turn()
        is_game_over = board.terminal()

    winner = board.winner()
    if winner == TURN_HUMAN:
        print("You win!")
    elif winner == TURN_AI:
        print("AI wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()