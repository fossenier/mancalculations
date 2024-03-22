#include <iostream>
#include <vector>

class MancalaBoard
{
public:
    /**
     * @brief Retrieves the active player.
     *
     * @return int The active player.
     */
    int getActivePlayer() const { return activePlayer; }
    /**
     * @brief Retrieves the number of stones in a specific pit for a given player.
     *
     * @param player The player (1 or 2) whose pit's stones are to be retrieved.
     * @param pit The number of the pit (1 to 6) for which the stone count is requested.
     * @return int The number of stones in the specified pit for the given player.
     */
    int getPlayerPit(int player, int pit) const
    {
        // reduce player and pit by 1 to get the correct index
        return playerPits[(player - 1) * 6 + (pit - 1)];
    }

    int getPlayerPit(int player, int pit) const { return playerPits[player * 6 + pit - 1]; }

private:
    std::vector<int> playerPits = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    std::vector<int> playerStores = {0, 0};
    int activePlayer = 0;
    bool is_game_over = false;
};

int main()
{
    std::vector<std::vector<int>> p_pits = {{4, 4, 4, 4, 4, 4}, {4, 4, 4, 4, 4, 4}};
    std::vector<int> p_store = {0, 0};

    [[maybe_unused]] bool is_game_over = false;
    [[maybe_unused]] int player_turn = 1;
}

/*
 """
    Runs a game of Mancala for the user.

    Args:
        `stdscr` (`stdscr`): Cureses main window.
    """
    # clear screen and hide cursor
    stdscr.clear()
    curses.curs_set(0)

    # setup initial game state
    board = MancalaBoard()
    is_game_over = False
    player_turn = 1
    board.draw_mancala_board(stdscr, VERTICAL_OFFSET_OF_BOARD, HORIZONTAL_OFFSET)

    # main game loop
    while not is_game_over:
        player_turn = 1 - player_turn
        board.draw_header(stdscr, player_turn)
        run_turn(stdscr, board, player_turn)
        is_game_over = board.check_game_over()

    # display game over message
    p1_score, p2_score = board.score_game()
    board.draw_game_over_animation(stdscr, p1_score, p2_score)
    */