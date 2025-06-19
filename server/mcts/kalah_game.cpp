#include "kalah_game.hpp"
#include <algorithm>
#include <numeric>

KalahGame::KalahGame() : board(14, 4), current_player(0), game_over(false)
{
    // Initialize board: 6 pits per player + 1 store per player
    // Stores are at indices 6 and 13
    board[6] = 0;  // Player 0's store
    board[13] = 0; // Player 1's store
}

KalahGame::KalahGame(const KalahGame &other)
    : board(other.board), current_player(other.current_player), game_over(other.game_over) {}

bool KalahGame::make_move(int pit)
{
    if (game_over || pit < 0 || pit >= 6 || board[pit + current_player * 7] == 0)
    {
        return false;
    }

    int current_pit = pit + current_player * 7;
    int seeds = board[current_pit];
    board[current_pit] = 0;

    // Distribute seeds
    while (seeds > 0)
    {
        current_pit = (current_pit + 1) % 14;

        // Skip opponent's store
        if ((current_player == 0 && current_pit == 13) ||
            (current_player == 1 && current_pit == 6))
        {
            continue;
        }

        board[current_pit]++;
        seeds--;
    }

    // Check for capture
    int player_store = current_player == 0 ? 6 : 13;
    if (current_pit >= current_player * 7 &&
        current_pit < current_player * 7 + 6 &&
        board[current_pit] == 1)
    {

        int opposite_pit = 12 - current_pit;
        if (board[opposite_pit] > 0)
        {
            board[player_store] += board[current_pit] + board[opposite_pit];
            board[current_pit] = 0;
            board[opposite_pit] = 0;
        }
    }

    check_game_over();

    // Extra turn if ended in own store
    if (current_pit == player_store)
    {
        return true;
    }

    switch_player();
    return false;
}

void KalahGame::check_game_over()
{
    // Check if either side is empty
    bool player0_empty = true;
    bool player1_empty = true;

    for (int i = 0; i < 6; i++)
    {
        if (board[i] > 0)
            player0_empty = false;
        if (board[i + 7] > 0)
            player1_empty = false;
    }

    if (player0_empty || player1_empty)
    {
        game_over = true;

        // Move remaining seeds to stores
        for (int i = 0; i < 6; i++)
        {
            board[6] += board[i];
            board[i] = 0;
            board[13] += board[i + 7];
            board[i + 7] = 0;
        }
    }
}

void KalahGame::switch_player()
{
    current_player = 1 - current_player;
}

bool KalahGame::is_game_over() const
{
    return game_over;
}

float KalahGame::get_reward(int player) const
{
    if (!game_over)
        return 0.0f;

    int player0_score = board[6];
    int player1_score = board[13];

    if (player == 0)
    {
        if (player0_score > player1_score)
            return 1.0f;
        else if (player0_score < player1_score)
            return -1.0f;
        else
            return 0.0f;
    }
    else
    {
        if (player1_score > player0_score)
            return 1.0f;
        else if (player1_score < player0_score)
            return -1.0f;
        else
            return 0.0f;
    }
}

std::vector<bool> KalahGame::get_valid_moves() const
{
    std::vector<bool> valid(6, false);

    if (game_over)
        return valid;

    int offset = current_player * 7;
    for (int i = 0; i < 6; i++)
    {
        valid[i] = board[offset + i] > 0;
    }

    return valid;
}

std::vector<float> KalahGame::get_canonical_state() const
{
    std::vector<float> state;
    state.reserve(15); // 14 board positions + 1 current player

    if (current_player == 0)
    {
        // Board as is
        for (int val : board)
        {
            state.push_back(static_cast<float>(val));
        }
    }
    else
    {
        // Swap perspectives
        for (int i = 7; i < 14; i++)
        {
            state.push_back(static_cast<float>(board[i]));
        }
        for (int i = 0; i < 7; i++)
        {
            state.push_back(static_cast<float>(board[i]));
        }
    }

    state.push_back(static_cast<float>(current_player));

    return state;
}
