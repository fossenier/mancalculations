"""
Handles the rules and logic of the game Mancala.
The human is the max player and the AI is the min player.
"""

from copy import deepcopy
from typing import List, Tuple

MAX_DEPTH = 11
PLAYER_PIT_COUNT = 6
TURN_HUMAN = 0
TURN_AI = 1


class Mancala(object):
    def __init__(self) -> None:
        self.__p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        self.__p_store = [0, 0]
        self.__p_turn = TURN_HUMAN

    def p_turn(self, turn: int = None) -> int:
        """
        Updates the current player turn if a value is provided.
        Otherwise, returns the current player turn.

        Raises a ValueError if the provided turn value is invalid.
        """
        if turn is not None:
            if turn not in [TURN_HUMAN, TURN_AI]:
                raise ValueError("Invalid turn value.")
        else:
            return self.__p_turn

    def actions(self) -> List[int]:
        """
        Returns a list of all possible actions (0 - 5) available to the current player.

        Raises a ValueError if the provided action is invalid.
        """
        return [pit for pit in range(6) if self.__p_pits[self.__p_turn][pit] != 0]

    def result(self, action: int) -> "Mancala":
        """
        Returns a new Mancala object that results from making the action on the board.

        Raises a ValueError if the provided action is invalid.
        """
        if action not in self.actions():
            raise ValueError("Invalid action.")

        new_board = deepcopy(self)
        new_board.__move_rocks(action)
        return new_board

    def winner(self) -> int:
        """
        Returns the winner of the game.
        0 for human, 1 for AI, -1 for a tie.
        None if the game is not over.
        """
        # there are still rocks in play
        if len(self.actions()) > 0:
            return None

        if self.__p_store[TURN_HUMAN] > self.__p_store[TURN_AI]:
            return TURN_HUMAN
        elif self.__p_store[TURN_AI] > self.__p_store[TURN_HUMAN]:
            return TURN_AI
        else:
            return -1

    def terminal(self) -> bool:
        """
        Returns True if the game is over.
        """
        return self.winner() is not None

    def utility(self) -> int:
        """
        Returns the utility of the game as the difference in stones scored.
        Positive for human, negative for AI, 0 for a tie.
        """
        # if there are no actions, add to the score the stones on the board
        if self.terminal():
            self.__p_store[TURN_HUMAN] += sum(self.__p_pits[TURN_HUMAN])
            self.__p_store[TURN_AI] += sum(self.__p_pits[TURN_AI])
        return self.__p_store[TURN_HUMAN] - self.__p_store[TURN_AI]

    def minimax(self) -> int:
        """
        Returns the optimal action for the current player.
        """

        def recursive_minimax(
            board: "Mancala", depth: int, alpha: int, beta: int
        ) -> Tuple[int, int]:
            """
            Uses minimax with alpha-beta pruning to determine the best move and utility for the current player,
            assuming optimal play.
            """
            if board.terminal() or depth == MAX_DEPTH:
                return board.utility(), None

            optimal_value = (
                float("-inf") if board.p_turn() == TURN_HUMAN else float("inf")
            )
            best_action = None

            for action in board.actions():
                new_board = board.result(action)
                value, _ = recursive_minimax(new_board, depth + 1, alpha, beta)

                if board.p_turn() == TURN_HUMAN:
                    if value > optimal_value:
                        optimal_value = value
                        best_action = action
                    alpha = max(alpha, value)
                else:
                    if value < optimal_value:
                        optimal_value = value
                        best_action = action
                    beta = min(beta, value)

                if alpha >= beta:
                    break

            return optimal_value, best_action

        # Start the recursion with initial values of alpha and beta
        return recursive_minimax(self, 0, float("-inf"), float("inf"))[1]

    def __move_rocks(self, action: int) -> str:
        """
        Moves the rocks from the selected pit.
        Attempts a steal if the last rock lands in an empty pit on the player's side.

        Will update self.__p_turn

        Raises a ValueError if the provided action is invalid.

        Returns
        "repeat" if the player gets another turn
        "steal" if a steal took place, and the player's turn is over
        "switch" if the player's turn is over
        """
        if action not in self.actions():
            raise ValueError("Invalid action.")

        rocks = self.__p_pits[self.__p_turn][action]
        self.__p_pits[self.__p_turn][action] = 0
        # track which side rocks are being placed
        active_side = self.__p_turn

        # place all picked up rocks one by one
        while rocks > 0:
            action += 1
            # at the end of the board
            if action == PLAYER_PIT_COUNT:
                # score a point if the last rock lands in the player's store
                if active_side == self.__p_turn:
                    self.__p_store[self.__p_turn] += 1
                    rocks -= 1
                # switch active side of the board, reset pit selection
                active_side = 1 - active_side
                action = -1
            # at any normal pit
            else:
                self.__p_pits[active_side][action] += 1
                rocks -= 1

        opposite_pit = PLAYER_PIT_COUNT - action - 1
        # the player gets another turn if the last rock lands in their store
        if active_side != self.__p_turn and action == -1:
            # do not update turn
            return "repeat"

        # a steal occurs if the last rock lands in an empty pit on the player's side
        # and there are rocks in the opposite pit
        elif (
            self.__p_pits[self.__p_turn][action] == 1
            and self.__p_pits[1 - self.__p_turn][opposite_pit] != 0
        ):

            self.__p_store[self.__p_turn] += self.__p_pits[1 - self.__p_turn][
                opposite_pit
            ]
            self.__p_store[self.__p_turn] += 1
            self.__p_pits[1 - self.__p_turn][opposite_pit] = 0
            self.__p_pits[self.__p_turn][action] = 0

            # update turn
            self.__p_turn = 1 - self.__p_turn
            return "steal"
        else:
            # update turn
            self.__p_turn = 1 - self.__p_turn
            return "switch"

    def __str__(self) -> str:
        """
        Returns a string representation of the current game state, emphasizing the circular arrangement.
        """
        # Assuming __p_pits is a list of lists where each sublist represents the pits for a player
        # and __p_store is a list of integers representing the store for each player
        pits_row1 = " ".join(f"{pit:02d}" for pit in self.__p_pits[0])
        pits_row2 = " ".join(
            f"{pit:02d}" for pit in self.__p_pits[1][::-1]
        )  # Reverse for visual alignment

        # Prepare the store and pits layout
        board_representation = (
            f"     {pits_row2}\n"
            f"{self.__p_store[1]:02d}                      {self.__p_store[0]:02d}\n"
            f"     {pits_row1}"
        )
        return board_representation
