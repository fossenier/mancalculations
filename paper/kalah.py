"""
Kalah game implementation.
Works in NDArray[np.float32] format for the neural network.
"""

import numpy as np
import numpy.typing as npt

class KalahGame:
    """
    Kalah game:
    - 6 pits each with 4 seeds + 1 store for each player
    - landing in an empty pit on your side captures opponent's seeds (if present)
    - landing in your store gives you another turn
    - game ends when one side is empty
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the class and resets its state.

        Calls the `reset` method to initialize or reset all relevant attributes.
        """
        self.reset()

    def reset(self) -> None:
        """
        Resets the game state to the initial configuration.

        Initializes the board with 4 stones in each pit and 0 in each store, sets the current player to
        Player 1 (0, south), clears the move history, marks the game as not over, and removes any winner.
        """
        # Board layout: [P1_pits(0-5), P1_store(6), P2_pits(7-12), P2_store(13)]
        self.board = np.array(
            [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0], dtype=np.float32  # Player 1
        )  # Player 2
        self.current_player = 0  # 0 for player 1 (South), 1 for player 2 (North)
        self.move_history = []
        self.game_over = False
        self.winner = None