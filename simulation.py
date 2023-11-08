"""
This is the backend code that determines the best move for the AI.
"""

import copy
from culations import Culations


class Simulation(Culations):
    def __init__(self):
        # basic game state
        # self.p_pits = [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]
        # smaller game state for testing
        self.p_pits = [[4, 4, 4, 0, 0, 0], [4, 4, 4, 0, 0, 0]]
        self.p_store = [0, 0]


def main():
    """
    Runs the simulation and spits out the SQL database with all appropriate moves.
    """
    total_game_ends = simulate_moves(Simulation(), 1, 0)
    print(total_game_ends)


def simulate_moves(board, player_turn, total):
    # save the state before making a move
    pre_state = ([board.p_pits[0].copy(), board.p_pits[1].copy()], board.p_store.copy())

    if board.check_game_over():
        return total + 1

    for pit in range(6):
        if board.p_pits[player_turn][pit] != 0:
            # make the move
            active_side, active_pit = board.move_rocks(player_turn, pit)

            # recurse based on the active pit
            if active_pit == -1:
                total = simulate_moves(board, player_turn, total)
            else:
                board.steal_rocks(active_side, player_turn, active_pit)
                total = simulate_moves(board, 1 - player_turn, total)
            # restore the state after making the move
            board.p_pits, board.p_store = pre_state
    return total


if __name__ == "__main__":
    main()
