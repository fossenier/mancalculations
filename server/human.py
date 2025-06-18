# human.py

import random
from kalah_game import KalahGame


def get_human_move(valid_moves):
    while True:
        try:
            move = int(input("Choose your move (0-5): "))
            if 0 <= move <= 5 and valid_moves[move] == 1:
                return move
            else:
                print("Invalid move. That pit is either empty or out of range.")
        except ValueError:
            print("Please enter an integer between 0 and 5.")


def get_random_move(valid_moves):
    valid_indices = [i for i, v in enumerate(valid_moves) if v == 1]
    return random.choice(valid_indices)


def main():
    game = KalahGame()

    print("Welcome to Kalah!")
    print("You are Player 1. The computer is Player 2.")
    game.render()

    while not game.game_over:
        valid_moves = game.get_valid_moves()

        if game.current_player == 0:
            print("Your turn.")
            move = get_human_move(valid_moves)
        else:
            move = get_random_move(valid_moves)
            print(f"Computer chose move: {move}")

        extra_turn = game.make_move(move)
        game.render()

        if extra_turn:
            print("Player gets another turn!")

    print("Game over!")
    print(f"Final Score - Player 1: {game.board[6]} | Player 2: {game.board[13]}")
    if game.winner == -1:
        print("It's a draw!")
    else:
        print(f"Player {game.winner + 1} wins!")


if __name__ == "__main__":
    main()
