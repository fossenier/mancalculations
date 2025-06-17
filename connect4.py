"""
Handles the rules and logic of the game Connect4.
The human is the max player and the AI is the min player.
"""

from copy import deepcopy
from typing import List, Tuple

MAX_DEPTH = 10  # Reduced depth due to complexity of Connect4
ROWS = 6
COLUMNS = 7
TURN_HUMAN = 0
TURN_AI = 1


class Connect4(object):
    def __init__(self) -> None:
        self.__board = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
        self.__p_turn = TURN_HUMAN

    def p_turn(self, turn: int = None) -> int:
        if turn is not None:
            if turn not in [TURN_HUMAN, TURN_AI]:
                raise ValueError("Invalid turn value.")
            self.__p_turn = turn
        else:
            return self.__p_turn

    def actions(self) -> List[int]:
        return [col for col in range(COLUMNS) if self.__board[0][col] == 0]

    def result(self, action: int) -> "Connect4":
        if action not in self.actions():
            raise ValueError("Invalid action.")
        new_board = deepcopy(self)
        for row in range(ROWS-1, -1, -1):
            if new_board.__board[row][action] == 0:
                new_board.__board[row][action] = self.__p_turn + 1
                break
        new_board.__p_turn = 1 - self.__p_turn
        return new_board

    def winner(self) -> int:
        for row in range(ROWS):
            for col in range(COLUMNS):
                if self.__check_winner_from_cell(row, col):
                    return self.__board[row][col] - 1
        if all(self.__board[0][col] != 0 for col in range(COLUMNS)):
            return -1
        return None

    def terminal(self) -> bool:
        return self.winner() is not None

    def     ty(self) -> int:
        if self.terminal():
            winner = self.winner()
            if winner == TURN_HUMAN:
                return 10000
            elif winner == TURN_AI:
                return -10000
            else:
                return 0  # Draw
        
        def count_windows(window, player):
            return window.count(player)
        
        score = 0
        # Score horizontal, vertical, and diagonal lines
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                window = [self.__board[row][col+i] for i in range(4)]
                if count_windows(window, 1) == 4:
                    score += 100
                elif count_windows(window, 2) == 4:
                    score -= 100
        
        for row in range(ROWS - 3):
            for col in range(COLUMNS):
                window = [self.__board[row+i][col] for i in range(4)]
                if count_windows(window, 1) == 4:
                    score += 100
                elif count_windows(window, 2) == 4:
                    score -= 100

        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                window = [self.__board[row+i][col+i] for i in range(4)]
                if count_windows(window, 1) == 4:
                    score += 100
                elif count_windows(window, 2) == 4:
                    score -= 100

        for row in range(ROWS - 3):
            for col in range(3, COLUMNS):
                window = [self.__board[row+i][col-i] for i in range(4)]
                if count_windows(window, 1) == 4:
                    score += 100
                elif count_windows(window, 2) == 4:
                    score -= 100
        
        return score

    def minimax(self) -> int:
        def recursive_minimax(board: "Connect4", depth: int, alpha: int, beta: int) -> Tuple[int, int]:
            if board.terminal() or depth == MAX_DEPTH:
                return board.utility(), None

            optimal_value = float("-inf") if board.p_turn() == TURN_HUMAN else float("inf")
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

        return recursive_minimax(self, 0, float("-inf"), float("inf"))[1]

    def __check_winner_from_cell(self, row: int, col: int) -> bool:
        if self.__board[row][col] == 0:
            return False

        def check_direction(delta_row: int, delta_col: int) -> bool:
            for i in range(1, 4):
                r, c = row + delta_row * i, col + delta_col * i
                if not (0 <= r < ROWS and 0 <= c < COLUMNS) or self.__board[r][c] != self.__board[row][col]:
                    return False
            return True

        return (
            check_direction(0, 1)  # Horizontal
            or check_direction(1, 0)  # Vertical
            or check_direction(1, 1)  # Diagonal down-right
            or check_direction(1, -1)  # Diagonal down-left
        )

    def __str__(self) -> str:
        board_representation = "\n".join(
            " ".join(f"{self.__board[row][col]}" for col in range(COLUMNS))
            for row in range(ROWS)
        )
        return board_representation