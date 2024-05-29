"""
Handles the rules and logic of the game Mancala.
"""

from copy import deepcopy
# from typing import 

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