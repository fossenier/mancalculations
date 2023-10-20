"""
This is the backend code for my Mancala game.
It provides all the functions needed to display the game.
"""

from constants import (
    HEIGHT_IDENTIFIERS,
    HEIGHT_PIT,
    HORIZONTAL_OFFSET,
    OFFSET_HEIGHT_PIT,
    OFFSET_WIDTH_PIT,
    PLAYER_PIT_COUNT,
    VERTICAL_OFFSET_OF_HEADER,
    P1,
    P2,
    WIDTH_PIT,
    WIDTH_STORE,
)
import curses as C


def draw_blank(stdscr, vertical_offset, horizontal_offset, lines):
    """
    Clears a specified quantity of lines.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `vertical_offset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
        `lines` (`integer`): Number of lines to clear.
    """
    # iterate through the lines and clear them
    for i in range(lines - 1):
        stdscr.move(vertical_offset + i, horizontal_offset)
        stdscr.clrtoeol()
    stdscr.refresh()


def draw_message(stdscr, vertical_offset, horizontal_offset, message):
    """
    Draws a message.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `vertical_offset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
        `message` (`string`): Message to display.
    """
    stdscr.addstr(vertical_offset, horizontal_offset, message)
    stdscr.refresh()


def draw_header(stdscr, player_turn):
    """
    Draws the header of the game.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `player_turn` (`integer`): Current player turn.\n
    """
    draw_message(stdscr, VERTICAL_OFFSET_OF_HEADER, HORIZONTAL_OFFSET, "Mancala")
    draw_message(
        stdscr,
        VERTICAL_OFFSET_OF_HEADER + 1,
        HORIZONTAL_OFFSET,
        f"Player {player_turn + 1}'s turn",
    )


def draw_mancala_board(stdscr, board, vertical_offfset, horizontal_offset):
    """
    Draws the current game state of the Mancala board.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
        `board` (`MancalaBoard`): Current game state.\n
        `vertical_offfset` (`integer`): Chosen Curses main window vertical offset.\n
        `horizontal_offset` (`integer`): Chosen Curses main window horizontal offset.\n
    """
    # draw Player 2's store (on the left below upper pit identifiers)
    draw_store(
        stdscr, board, P1, vertical_offfset + HEIGHT_IDENTIFIERS, horizontal_offset
    )
    horizontal_offset += WIDTH_STORE
    for i in range(PLAYER_PIT_COUNT):
        # draw Player 2's pit identifiers (on the top and in reverse order)
        draw_pit_identifier(
            stdscr,
            vertical_offfset,
            # TODO change this to PIT_WIDTH - 1 when I change the pit width
            horizontal_offset + i * OFFSET_WIDTH_PIT + OFFSET_WIDTH_PIT // 2,
            PLAYER_PIT_COUNT - i,
        )
        # draw Player 2's pits (in the middle below upper pit identifiers in reverse order)
        draw_pit(
            stdscr,
            board,
            P2,
            vertical_offfset + HEIGHT_IDENTIFIERS,
            # TODO change this to PIT_WIDTH - 1 when I change the pit width
            horizontal_offset
            + (PLAYER_PIT_COUNT - i) * OFFSET_WIDTH_PIT
            - OFFSET_WIDTH_PIT // 2,
            i,
        )
        # draw Player 1's pits (in the middle above lower pit identifiers)
        draw_pit(
            stdscr,
            board,
            P1,
            vertical_offfset + HEIGHT_IDENTIFIERS + OFFSET_WIDTH_PIT // 2,
            # TODO change this to PIT_WIDTH - 1 when I change the pit width
            horizontal_offset + i * OFFSET_WIDTH_PIT + OFFSET_WIDTH_PIT // 2,
            i,
        )
        # draw Player 2's pit identifiers (on the bottom)
        draw_pit_identifier(
            stdscr,
            vertical_offfset + HEIGHT_IDENTIFIERS + HEIGHT_PIT + OFFSET_HEIGHT_PIT,
            # TODO change this to PIT_WIDTH - 1 when I change the pit width
            horizontal_offset + i * OFFSET_WIDTH_PIT + OFFSET_WIDTH_PIT // 2,
            i + 1,
        )
    # shift over 1 to the right for the final pit's closing |
    horizontal_offset += 1
    # draw Player 1's store (on the right above lower pit identifiers)
    draw_store(
        stdscr,
        board,
        P1,
        vertical_offfset + HEIGHT_IDENTIFIERS,
        horizontal_offset + OFFSET_WIDTH_PIT * PLAYER_PIT_COUNT,
    )
    stdscr.refresh()


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
        "-" * (WIDTH_PIT + 1),
        f"{board.p_pits[player_turn][pit_selection]:^{WIDTH_PIT}}|",
        "-" * (WIDTH_PIT + 1),
    ]

    line_count = 0
    for line in lines_to_draw:
        stdscr.addstr(vertical_offset + line_count, horizontal_offset, line)
        line_count += 1


def draw_pit_identifier(stdscr, vertical_offset, horizontal_offset, pit_selection):
    stdscr.addstr(vertical_offset, horizontal_offset, f"{pit_selection:^{WIDTH_PIT}}")


def draw_pit_selection(stdscr, vertical_offset, horizontal_offset):
    """
    Draws in the terminal asking the user to select a pit.

    Args:
        `stdscr` (`stdscr`): Curses main window.\n
    """
    # prompt user to enter a valid pit
    stdscr.addstr(vertical_offset, horizontal_offset, "Choose a valid pit (1-6): ")
    stdscr.refresh()

    # get the user's input
    C.echo()
    user_input = stdscr.getstr().decode("utf-8")
    C.noecho()
    return user_input


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
        "-" * (WIDTH_STORE + 2),
        f"|{'':^{WIDTH_STORE}}|",
        f"|{board.p_store[player_turn]:^{WIDTH_STORE}}|",
        f"|{'':^{WIDTH_STORE}}|",
        "-" * (WIDTH_STORE + 2),
    ]

    for line in lines_to_draw:
        stdscr.addstr(vertical_offfset, horizontal_offset, line)
        vertical_offfset += 1
