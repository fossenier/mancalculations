"""
Kalah (6,4) game implementation with efficient state representation
"""

import numpy as np
from typing import List, Tuple, Optional
import copy


class KalahGame:
    """
    Kalah game implementation following standard rules:
    - 6 pits per player with 4 stones each initially
    - Capture opposite pit if landing in own empty pit
    - Extra turn if landing in own store
    - Game ends when one side is empty
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset game to initial state"""
        # Board layout: [P1_pits(0-5), P1_store(6), P2_pits(7-12), P2_store(13)]
        self.board = np.array(
            [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0], dtype=np.int32  # Player 1
        )  # Player 2
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.move_history = []
        self.game_over = False
        self.winner = None

    def get_state(self) -> np.ndarray:
        """Get current board state from current player's perspective"""
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

    def get_canonical_state(self) -> np.ndarray:
        """Get board state normalized by total stones"""
        state = self.get_state()
        total_stones = np.sum(state)
        if total_stones > 0:
            return state.astype(np.float32) / total_stones
        return state.astype(np.float32)

    def get_valid_moves(self) -> np.ndarray:
        """Get binary mask of valid moves"""
        valid = np.zeros(6, dtype=np.float32)
        if self.current_player == 0:
            valid[:6] = (self.board[:6] > 0).astype(np.float32)
        else:
            valid[:6] = (self.board[7:13] > 0).astype(np.float32)
        return valid

    def make_move(self, action: int) -> bool:
        """
        Make a move and return True if player gets another turn
        action: pit index (0-5) from current player's perspective
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
                opposite = 12 - (current_pos - 7)
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

    def get_reward(self, player: int) -> float:
        """Get reward from perspective of given player"""
        if not self.game_over:
            return 0.0

        if self.winner == -1:  # Draw
            return 0.0
        elif self.winner == player:
            return 1.0
        else:
            return -1.0

    def get_score_difference(self) -> int:
        """Get score difference (P1 - P2)"""
        return self.board[6] - self.board[13]

    def clone(self):
        """Create a deep copy of the game"""
        new_game = KalahGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def render(self):
        """Pretty print the board"""
        print("\n" + "=" * 50)
        print(f"Player 2: {' '.join(f'{self.board[12-i]:2d}' for i in range(6))}")
        print(f"Stores:  {self.board[13]:2d}" + " " * 34 + f"{self.board[6]:2d}")
        print(f"Player 1: {' '.join(f'{self.board[i]:2d}' for i in range(6))}")
        print(f"Current player: {self.current_player + 1}")
        print("=" * 50 + "\n")

    def get_symmetries(self, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all symmetries of the board and policy"""
        # Kalah board has no symmetries due to directional play
        return [(self.get_canonical_state(), pi)]

    def __str__(self):
        return f"KalahGame(player={self.current_player}, over={self.game_over})"
