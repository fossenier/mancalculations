"""
Handles the rules and logic of the game Mancala.
"""

from copy import deepcopy
from typing import List, Tuple

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
        elif (self.__p_pits[self.__p_turn][action] == 1
            and self.__p_pits[1 - self.__p_turn][opposite_pit] != 0):
            
            self.__p_store[self.__p_turn] += self.__p_pits[1 - self.__p_turn][opposite_pit]
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