"""
This is the list of constants for my Mancala game.
"""

# Toggle-able constants
HORIZONTAL_LINE = "-"  # any single character, draws the horizontal lines
HORIZONTAL_OFFSET = 0  # any integer, shifts over the board
MESSAGE_WAIT = 1.5  # any positive number, controls how long messages are displayed
OFFSET_WIDTH_PIT = 4  # all even numbers 2 and up, controls the width of the pits
VERTICAL_LINE = "|"  # any single character, draws the vertical lines
VERTICAL_OFFSET_OF_BOARD = 4  # shifts the board down, takes up 7 lines
VERTICAL_OFFSET_OF_ERROR = 13  # shifts the error message down, takes up 1 line
VERTICAL_OFFSET_OF_HEADER = 0  # shifts the header down, takes up 2 lines
VERTICAL_OFFSET_OF_PROMPT = 12  # shifts the prompt down, takes up 1 line
WIDTH_STORE = 7  # all odd numbers 3 and up, controls the width of the stores


# Dependant constants
WIDTH_PIT = OFFSET_WIDTH_PIT + 1


# Non-toggle-able constants
HEIGHT_IDENTIFIERS = 1  # why? the identifiers are hard-coded as being a row of 1 number
EFFECTIVE_HEIGHT_PIT = 2  # why? the pits are hard-coded with two lines, and a number
P1 = 0  # why? the player identifiers are hard-coded as 0 and 1 for indexing
P2 = 1  # why? the player identifiers are hard-coded as 0 and 1 for indexing
PLAYER_PIT_COUNT = 6  # why? the pits are hard-coded as being 6 per player
VALID_USER_PITS = [1, 2, 3, 4, 5, 6]  # why? the player can only has 6 pits
