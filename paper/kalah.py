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
            [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0], dtype=np.float32
        )
        self.current_player = 0  # 0 for player 1 (South), 1 for player 2 (North)
        self.game_over = False
        self.winner = None

    def get_state(self) -> npt.NDArray[np.float32]:
        """
        Returns the current state of the game as a 14-element array.
        The state includes the number of seeds in each pit and store.
        """
        return self.board.copy()

    def get_canonical_state(self) -> npt.NDArray[np.float32]:
        """
        Returns the canonical state of the game.
        The canonical state is the current state from the perspective of the
        player who is next to take an action.
        """
        if self.current_player == 1:
            return np.concatenate((self.board[7:14], self.board[0:7])).copy()
        return self.board.copy()

    def get_valid_moves(self) -> npt.NDArray[np.float32]:
        """
        Returns a binary array of 6 elements indicating valid moves for the
        current player. A move is valid if the selected pit has seeds.
        """
        # Initialize all to zero (invalid)
        valid_moves = np.zeros(6, dtype=np.float32)
        start_index = 0 if self.current_player == 0 else 7
        for i in range(6):
            if self.board[start_index + i] > 0:
                valid_moves[i] = 1.0
        return valid_moves

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

    def get_reward(self, player: int | None = None) -> float:
        """
        Returns the reward for the specified player. Defaults to active player.
        Positive reward for winning, negative for losing, and zero for a draw.
        """
        # Default to current player
        if player is None or player not in [0, 1]:
            player = self.current_player

        if not self.game_over:
            raise ValueError("Game is not over yet, cannot determine reward.")
        if self.winner == player:
            return 1.0
        elif self.winner == -1:
            return 0.0
        else:
            return -1.0

    def clone(self) -> "KalahGame":
        """
        Creates a deep copy of the current game state.

        Returns:
            KalahGame: A new instance of KalahGame with the same state as the current game.
        """
        new_game = KalahGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def __str__(self) -> str:
        """
        Displays the current state of the Kalah game board in a formatted manner.

        The board is represented with Player 2's pits at the top (indices 12 to 7), both players' stores,
        Player 1's pits at the bottom (indices 0 to 5), and indicates the current player's turn.

        Returns:
            None
        """
        board_str = ""
        for i in range(12, 6, -1):
            board_str += f" {self.board[i]} "
        board_str += f"\n{self.board[13]} : {self.board[6]}\n"
        for i in range(0, 6):
            board_str += f" {self.board[i]} "
        board_str += f"\nCurrent Player: {self.current_player + 1}"
        return board_str
