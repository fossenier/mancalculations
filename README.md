# Mancalculations

## Description

This is a Python implementation of Mancala with AI capabilities. The game logic is handled in Python, with a command-line interface for user interaction. The game uses the minimax algorithm with alpha-beta pruning to determine optimal moves for the AI.

## Table of Contents

- [Mancala](#mancala)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Implementation](#implementation)
  - [1. Game Setup](#1-game-setup)
  - [2. Game Logic](#2-game-logic)
  - [3. AI Design Choices](#3-ai-design-choices)
  - [4. User Interaction](#4-user-interaction)
- [Configuration](#configuration)
- [Credits](#credits)
- [Changelog](#changelog)

## Mancala:

For those of you unfamiliar with Mancala, it is a children's game played on pits with stones. Each player has six pits in front of them, each with 4 stones. To their right is their store, where they collect points. Every turn, the player picks up all stones in a non-empty pit. They drop one stone in each pit going counterclockwise, including scoring a point in their store, and including each pit on the opponent's side (but not their store!) continuing until they are out of stones. If they land in their store, they can move again. If they land in an empty pit on their side where the opponent's pit immediately opposite has stones, they pick up that last stone and the opponent's stones opposite and collect them in their store. Once a player cannot take another move, both players add up their store stones and any stones on their side of the board for the final tabulation.

## Prerequisites

Written in Python 3.12.3, though it may work in versions to come / versions gone by.

## Usage

- Run `runner.py` and the game will execute
- The game will first display the board in it's initial state with the player side below and the AI side above, it will prompt with "AI plays x" and ask "Enter your move:" to take your input from 0 to 6

## Implementation:

The implementation of Mancala in Python can be broken down as follows:

### 1. Game Setup:

The `Mancala` class initializes the game board with the standard setup of 6 pits per player, each containing 4 stones, and 2 stores starting with 0 stones.

### 2. Game Logic:

The game logic includes methods to handle player turns, move stones, and check for game termination. The main methods are:

- `p_turn()`: Returns or updates the current player's turn.
- `actions()`: Returns a list of valid actions for the current player.
- `result()`: Returns a new game state after a given action.
- `winner()`: Determines the winner of the game.
- `terminal()`: Checks if the game is over.
- `utility()`: Calculates the utility of the game state.
- `minimax()`: Implements the minimax algorithm with alpha-beta pruning to determine the best move for the AI.

### 3. AI Design Choices:

The AI uses the minimax algorithm with alpha-beta pruning to optimize its moves. The depth of the search tree is controlled by `MAX_DEPTH` to balance between performance and decision quality. Here are the key design choices:

- **Minimax Algorithm**: Used for decision-making to simulate all possible moves and counter-moves, aiming to maximize the human player's score and minimize the AI's score.
- **Alpha-Beta Pruning**: Enhances the minimax algorithm by pruning branches that won't affect the final decision, improving efficiency.
- **Depth Limitation**: Set to a maximum depth (`MAX_DEPTH = 12`) to ensure the AI makes decisions in a reasonable time frame.

### 4. User Interaction:

The game runs in a command-line interface, allowing the user to play against the AI. The main interaction loop:

- Displays the current game board.
- Prompts the human player for their move.
- Executes the AI's move using the minimax algorithm.
- Continues until the game ends, displaying the final score and winner.

## Configuration

- Inside of `mancala.py`, feel free to modify self.\_\_p_turn to determine who goes first, or modify MAX_DEPTH to suit your device (more depth takes more resources)

## Credits

- I was inspired by a long time friend from Scouts Canada after he showed me Mancala, and the final iteration of the game came after learning about Minimax in Harvard's CS50 AI course online

## Changelog

- Version history
