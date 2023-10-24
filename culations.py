"""
This is backend code for my Mancala game.
It provides all the functions needed to run the game.
"""

from constants import PLAYER_PIT_COUNT


class Culations:
    def check_game_over(self):
        """
        Checks if one side's pits are empty.

        Args:
            `self` (`MancalaBoard`): Current game state.

        Returns:
            `boolean`: Whether or not the game is over.
        """
        # check if the game is over
        for player in range(2):
            if sum(self.p_pits[player]) == 0:
                return True
        return False

    def move_rocks(self, player_turn, pit_selection):
        """
        Moves the rocks from the selected pit.

        Args:
            `self` (`MancalaBoard`): Current game state.\n
            `player_turn` (`integer`): Current player turn.\n
            `pit_selection` (`integer`): Player's selected pit of choice.

        Returns:
            `player_turn` (`integer`): Most recent active player's side of the board.\n
            `pit_selection` (`integer`): Most recent active pit.
        """
        # track active side of the board, pick up rocks from selected pit
        active_player = player_turn
        rock_count, self.p_pits[active_player][pit_selection] = (
            self.p_pits[active_player][pit_selection],
            0,
        )
        # move rocks from selected pit
        while rock_count > 0:
            pit_selection += 1
            if pit_selection == PLAYER_PIT_COUNT:
                # add to store if the active side matches the player's turn
                if active_player == player_turn:
                    self.p_store[active_player] += 1
                    rock_count -= 1
                # switch active side of the board, reset pit selection
                active_player = 1 - active_player
                pit_selection = -1
            else:
                self.p_pits[active_player][pit_selection] += 1
                rock_count -= 1
        return active_player, pit_selection

    def score_game(self):
        """
        Calculates the final score of the game.

        Args:
            self (`MancalaBoard`): Current game state.

        Returns:
            `integer`: Player 1's final score.\n
            `integer`: Player 2's final score.
        """
        # add up the score
        p1_score = self.p_store[0] + sum(self.p_pits[0])
        p2_score = self.p_store[1] + sum(self.p_pits[1])
        return p1_score, p2_score

    def steal_rocks(self, active_side, player_turn, active_pit):
        """
        Performs a steal if the last rock landed in an empty pit on the player's side.

        Args:
            `self` (`MancalaBoard`): Current game state.\n
            `active_side` (`integer`): Most recent active side of the board.\n
            `player_turn` (`integer`): Current player turn.\n
            `active_pit` (`integer`): Most recent active pit.

        Returns:
            `is_steal` (`boolean`): Whether or not a steal occurred.
        """
        if active_side != player_turn:
            return False
        is_steal = False
        # check if the last rock landed in an empty pit on the player's side
        if (
            self.p_pits[active_side][active_pit] == 1
            and self.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit] != 0
        ):
            is_steal = True
            # steal the rocks from the opposite pit
            self.p_store[active_side] += (
                self.p_pits[active_side][active_pit]
                + self.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit]
            )
            self.p_pits[active_side][active_pit] = 0
            self.p_pits[1 - active_side][(PLAYER_PIT_COUNT - 1) - active_pit] = 0
        return is_steal
