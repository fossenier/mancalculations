"""
Kalah (6,4) game implementation with efficient state representation
"""

import numpy as np
import numpy.typing as npt
from typing import List, Tuple


class KalahGame:
    """
    Kalah game implementation following standard rules:
    - 6 pits per player with 4 stones each initially
    - Capture opposite pit if landing in own empty pit
    - Extra turn if landing in own store
    - Game ends when one side is empty
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

        Initializes the board with 4 stones in each pit and 0 in each store, sets the current player to Player 1 (South),
        clears the move history, marks the game as not over, and removes any winner.
        """
        # Board layout: [P1_pits(0-5), P1_store(6), P2_pits(7-12), P2_store(13)]
        self.board = np.array(
            [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0], dtype=np.int32  # Player 1
        )  # Player 2
        self.current_player = 0  # 0 for player 1 (South), 1 for player 2 (North)
        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_state(self) -> npt.NDArray[np.int32]:
        """
        Returns the current state of the board as a NumPy array from the perspective of the current player.

        If the current player is player 0, returns a copy of the board as is.
        If the current player is player 1, rotates the board so that player 1's pits and store appear in the same positions as player 0's perspective.

        Returns:
            np.ndarray: The board state as a NumPy array of int32, oriented for the current player.
        """
        if self.current_player == 0:
            return self.board.copy()
        else:
            # Rotate board for player 2's perspective
            return np.concatenate(
                [
                    self.board[7:14],  # P2's side becomes P1's side
                    self.board[0:7],  # P1's side becomes P2's side
                ]
            )

    def get_canonical_state(self) -> npt.NDArray[np.int32]:
        """
        Returns a canonical representation of the game state as a NumPy array, excluding the stores.
        The canonical state consists of the pits for both players, omitting the stores.

        Returns:
            np.ndarray: A 1D NumPy array of type int32 containing the pit values for both players, excluding the stores.
        """
        state = self.get_state()
        # Ignore the stores at indices 6 and 13
        return np.concatenate([state[:6], state[7:13]])

    def get_valid_moves(self) -> np.ndarray:
        """
        Returns an array indicating the valid moves for the current player.

        A move is considered valid if the corresponding pit contains at least one stone.
        For player 0, pits 0-5 are checked; for player 1, pits 7-12 are checked.
        The returned array has length 6, where each element is 1 if the move is valid, 0 otherwise.

        Returns:
            np.ndarray: A 1D array of shape (6,) with 1s for valid moves and 0s for invalid moves.
        """
        valid = np.zeros(6, dtype=np.int32)
        # Mark down non-empty pits as valid
        if self.current_player == 0:
            valid[:6] = (self.board[:6] > 0).astype(np.int32)
        else:
            valid[:6] = (self.board[7:13] > 0).astype(np.int32)
        return valid

    def make_move(self, action: int) -> bool:
        """
        Executes a move for the current player in the Kalah game.

        Args:
            action (int): The pit index (0-5) from which the current player wants to move stones.

        Returns:
            bool: True if the player gets an extra turn (landed in their own store), False otherwise.

        Raises:
            ValueError: If the action is out of bounds or the selected pit is empty.

        Side Effects:
            - Updates the game board according to the rules of Kalah.
            - Handles stone distribution, captures, and extra turns.
            - Updates move history.
            - Checks for and handles game end, collecting remaining stones and determining the winner.
            - Switches the current player if no extra turn is granted.
        """
        if action < 0 or action > 5:
            raise ValueError(f"Invalid action: {action}")

        # Convert to absolute board position
        if self.current_player == 0:
            pit = action
        else:
            pit = action + 7

        # Check if move is valid
        if self.board[pit] == 0:
            raise ValueError(f"Cannot move from empty pit: {action}")

        # Pick up stones
        stones = self.board[pit]
        self.board[pit] = 0

        # Distribute stones
        current_pos = pit
        while stones > 0:
            current_pos = (current_pos + 1) % 14

            # Skip opponent's store
            if (self.current_player == 0 and current_pos == 13) or (
                self.current_player == 1 and current_pos == 6
            ):
                continue

            self.board[current_pos] += 1
            stones -= 1

        # Check for capture
        landed_in_store = False
        if self.current_player == 0:
            if 0 <= current_pos <= 5 and self.board[current_pos] == 1:
                opposite = 12 - current_pos
                if self.board[opposite] > 0:
                    self.board[6] += self.board[current_pos] + self.board[opposite]
                    self.board[current_pos] = 0
                    self.board[opposite] = 0
            landed_in_store = current_pos == 6
        else:
            if 7 <= current_pos <= 12 and self.board[current_pos] == 1:
                opposite = 12 - current_pos
                if self.board[opposite] > 0:
                    self.board[13] += self.board[current_pos] + self.board[opposite]
                    self.board[current_pos] = 0
                    self.board[opposite] = 0
            landed_in_store = current_pos == 13

        # Record move
        self.move_history.append((self.current_player, action))

        # Check for game end
        if np.sum(self.board[0:6]) == 0 or np.sum(self.board[7:13]) == 0:
            # Collect remaining stones
            self.board[6] += np.sum(self.board[0:6])
            self.board[13] += np.sum(self.board[7:13])
            self.board[0:6] = 0
            self.board[7:13] = 0

            # Determine winner
            self.game_over = True
            if self.board[6] > self.board[13]:
                self.winner = 0
            elif self.board[13] > self.board[6]:
                self.winner = 1
            else:
                self.winner = -1  # Draw

        # Switch player if no extra turn
        if not landed_in_store and not self.game_over:
            self.current_player = 1 - self.current_player

        return landed_in_store

    def get_reward(self, player: int = -1) -> float:
        """
        Calculates and returns the reward for the specified player based on the game outcome.

        Args:
            player (int, optional): The player for whom to calculate the reward (0 or 1).
                If not specified or invalid, defaults to the current player.

        Returns:
            float: The reward value:
                - 1.0 if the specified player has won,
                - -1.0 if the specified player has lost,
                - 0.0 if the game is a draw or not yet over.
        """
        # Default to current player
        if player not in [0, 1]:
            player = self.current_player

        if not self.game_over:
            return 0.0

        if self.winner == -1:  # Draw
            return 0.0
        elif self.winner == player:
            return 1.0
        else:
            return -1.0

    def get_score_difference(self) -> int:
        """
        Calculates and returns the score difference between the two players.

        Returns:
            int: The difference between the current player's score and the
            opponent's score on the board.
        """
        active_player_store = 6 if self.current_player == 0 else 13
        opponent_store = 13 if self.current_player == 0 else 6
        # Return the difference between the current player's store and the opponent's store
        return self.board[active_player_store] - self.board[opponent_store]

    def clone(self) -> "KalahGame":
        """
        Creates and returns a deep copy of the current KalahGame instance.

        Returns:
            KalahGame: A new instance of KalahGame with the same board state,
            current player, move history, game over status, and winner as the original.
        """
        new_game = KalahGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def render(self):
        """
        Displays the current state of the Kalah game board in a formatted manner.

        The board is printed with Player 2's pits at the top (indices 12 to 7), both players' stores,
        Player 1's pits at the bottom (indices 0 to 5), and indicates the current player's turn.

        Returns:
            None
        """
        print("\n" + "=" * 50)
        print(f"Player 2: {' '.join(f'{self.board[12-i]:2d}' for i in range(6))}")
        print(f"Stores:  {self.board[13]:2d}" + " " * 34 + f"{self.board[6]:2d}")
        print(f"Player 1: {' '.join(f'{self.board[i]:2d}' for i in range(6))}")
        print(f"Current player: {self.current_player + 1}")
        print("=" * 50 + "\n")

    def __str__(self):
        return f"KalahGame(player={self.current_player}, over={self.game_over})"
