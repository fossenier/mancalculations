"""
This is the backend code for my Mancala game.
It provides all the functions needed to display the game.
"""

from constants import PIT_WIDTH, PLAYER_PIT_COUNT, STORE_WIDTH


def draw_footer_blank(stdscr, vertical_offset, horizontal_offset):
    """
    Clears the footer in the terminal.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
    """
    for i in range(5):
        stdscr.move(vertical_offset + i, horizontal_offset)
        stdscr.clrtoeol()
    stdscr.refresh()


def draw_footer_pit_selection(stdscr, vertical_offset, horizontal_offset):
    """
    Draws a footer in the terminal asking the user to select a pit.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
    """
    stdscr.addstr(vertical_offset, horizontal_offset, "Choose a valid pit (1-6): ")
    stdscr.refresh()

    # Get the user's input
    user_input = stdscr.getstr().decode("utf-8")

    # Convert the input to an integer
    try:
        pit_selection = int(user_input)
        if pit_selection < 1 or pit_selection > 6:
            # Handle invalid range
            stdscr.addstr(
                vertical_offset + 1,
                horizontal_offset,
                "Invalid choice. Please enter a number between 1 and 6.",
            )
            stdscr.refresh()
            draw_footer_pit_selection(stdscr, vertical_offset, horizontal_offset)
        else:
            stdscr.move(vertical_offset + 1, horizontal_offset)
            stdscr.clrtoeol()
            stdscr.refresh()
    except ValueError:
        # Handle non-integer input
        stdscr.addstr(
            vertical_offset + 1,
            horizontal_offset,
            "Invalid input. That wasn't a number.",
        )
        stdscr.refresh()

    draw_footer_blank(stdscr, vertical_offset, horizontal_offset)
    return pit_selection


def draw_header(stdscr, player_turn):
    """
    Draws the header of the game in the terminal.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `player_turn` (`integer`): Current player turn.\n
    """
    stdscr.addstr(0, 0, "Mancala Game")
    stdscr.addstr(1, 0, f"Player {player_turn + 1}'s turn")


def draw_mancala_board(stdscr, board, vertical_offfset, horizontal_offset):
    """
    Draws the current game state of the Mancala board in the terminal.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
    """
    # Draw Player 2's store (on the left)
    draw_store(stdscr, board, 1, vertical_offfset, horizontal_offset)
    horizontal_offset += 3
    # For each player
    for i in range(2):
        # Draw each pit
        for j in range(PLAYER_PIT_COUNT):
            draw_pit(
                stdscr,
                board,
                i,
                vertical_offfset + i * 2,
                horizontal_offset + (j + 1) * 4,
                j,
            )
    # Draw Player 1's store (on the right)
    draw_store(
        stdscr, board, 0, vertical_offfset, horizontal_offset + PLAYER_PIT_COUNT * 4 - 1
    )


def draw_pit(
    stdscr, board, player_turn, vertical_offset, horizontal_offset, pit_selection
):
    """
    Draws a player Mancala pit at the given offset location.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
        `pit_selection` (`integer`): Chosen player pit.\n"""
    lines_to_draw = [
        "-" * (PIT_WIDTH + 1),
        f"{board.p_pits[player_turn][pit_selection]:^{PIT_WIDTH}}|",
        "-" * (PIT_WIDTH + 1),
    ]

    for line in lines_to_draw:
        stdscr.addstr(vertical_offset, horizontal_offset, line)
        vertical_offset += 1


def draw_store(stdscr, board, player_turn, vertical_offfset, horizontal_offset):
    """
    Draws a player Mancala store at the given offset location.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `player_turn` (`integer`): Current player turn.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
    """
    lines_to_draw = [
        "-" * (STORE_WIDTH + 2),
        f"|{'':^{STORE_WIDTH}}|",
        f"|{board.p_store[player_turn]:^{STORE_WIDTH}}|",
        f"|{'':^{STORE_WIDTH}}|",
        "-" * (STORE_WIDTH + 2),
    ]

    for line in lines_to_draw:
        stdscr.addstr(vertical_offfset, horizontal_offset, line)
        vertical_offfset += 1
