#pragma once

#include <vector>
#include <memory>

class KalahGame
{
public:
    KalahGame();
    KalahGame(const KalahGame &other);

    // Core game functions
    bool make_move(int pit);
    bool is_game_over() const;
    float get_reward(int player) const;
    std::vector<bool> get_valid_moves() const;
    std::vector<float> get_canonical_state() const;

    // Utility functions
    int get_current_player() const { return current_player; }
    KalahGame clone() const { return KalahGame(*this); }

private:
    std::vector<int> board;
    int current_player;
    bool game_over;

    void check_game_over();
    void switch_player();
};


#pragma once

#include <vector>
#include <memory>
#include <utility>

class KalahGame
{
public:
    KalahGame();
    KalahGame(const KalahGame &other);

    // Core game functions
    bool make_move(int pit);
    bool is_game_over() const;
    float get_reward(int player = -1) const;
    std::vector<bool> get_valid_moves() const;
    std::vector<float> get_canonical_state() const;

    // Utility functions
    int get_current_player() const { return current_player; }
    KalahGame clone() const { return KalahGame(*this); }
    int get_score_difference() const;

private:
    std::vector<int> board;
    int current_player;
    bool game_over;
    int winner; // -1 for draw, 0 for player 0, 1 for player 1, -2 for no winner yet
    std::vector<std::pair<int, int>> move_history; // (player, action) pairs

    void check_game_over();
    void switch_player();
};