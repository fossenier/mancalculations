#!/usr/bin/env python3
"""
Test script for MCTS implementation with performance monitoring and debugging capabilities.
Run from console to observe MCTS behavior, performance metrics, and debug issues.
"""

import time
import numpy as np
from multiprocessing import Queue
from typing import Dict, List, Tuple
import argparse
import sys
from collections import defaultdict
from config import AlphaZeroConfig

CONFIG = AlphaZeroConfig()

# Import your modules (adjust paths as needed)
try:
    from mcts import MCTS, MCTSStatistic
    from kalah import KalahGame
    from config import AlphaZeroConfig
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure mcts.py, kalah.py, and config.py are in the same directory or PYTHONPATH"
    )
    sys.exit(1)


class MockModel:
    """Mock neural network model for testing MCTS without actual GPU inference."""

    def __init__(self, randomness: float = 0.1):
        self.randomness = randomness
        self.call_count = 0
        self.call_times = []

    def predict(self, state) -> Tuple[np.ndarray, float]:
        """Return mock policy and value predictions."""
        self.call_count += 1
        start_time = time.time()

        # Mock some computation time
        # time.sleep(0.001)  # 1ms mock inference time

        # Generate semi-realistic policy (favor middle pits slightly)
        policy = np.random.rand(6).astype(np.float32)
        # Add slight bias toward middle positions
        policy[2:4] *= 1.2
        policy = policy / policy.sum()  # Normalize

        # Add some randomness
        if np.random.rand() < self.randomness:
            policy = np.random.rand(6).astype(np.float32)
            policy = policy / policy.sum()

        # Mock value between -1 and 1
        value = np.float32(np.random.rand() * 2 - 1)

        self.call_times.append(time.time() - start_time)
        return policy, float(value)


class MCTSMonitor:
    """Monitor and debug MCTS performance."""

    def __init__(self):
        self.stats = {
            "games_completed": 0,
            "total_moves": 0,
            "model_calls": 0,
            "avg_game_length": 0.0,
            "start_time": time.time(),
            "move_times": [],
            "simulation_counts": [],
            "tree_depths": [],
        }
        self.game_histories = []

    def log_move(self, move_time: float, simulations: int, tree_depth: int = 0):
        """Log statistics for a single move."""
        self.stats["total_moves"] += 1
        self.stats["move_times"].append(move_time)
        self.stats["simulation_counts"].append(simulations)
        self.stats["tree_depths"].append(tree_depth)

    def log_game_complete(self, game_history: List[MCTSStatistic]):
        """Log completion of a game."""
        self.stats["games_completed"] += 1
        self.game_histories.append(game_history)
        if self.stats["games_completed"] > 0:
            self.stats["avg_game_length"] = (
                self.stats["total_moves"] / self.stats["games_completed"]
            )

    def print_stats(self):
        """Print current performance statistics."""
        elapsed = time.time() - self.stats["start_time"]

        print(f"\n{'='*60}")
        print(f"MCTS PERFORMANCE STATS")
        print(f"{'='*60}")
        print(f"Runtime: {elapsed:.2f}s")
        print(f"Games completed: {self.stats['games_completed']}")
        print(f"Total moves: {self.stats['total_moves']}")
        print(f"Model calls: {self.stats['model_calls']}")
        print(f"Avg game length: {self.stats['avg_game_length']:.1f} moves")

        if self.stats["move_times"]:
            print(f"Avg move time: {np.mean(self.stats['move_times']):.3f}s")
            print(f"Max move time: {np.max(self.stats['move_times']):.3f}s")

        if self.stats["simulation_counts"]:
            print(
                f"Avg simulations/move: {np.mean(self.stats['simulation_counts']):.1f}"
            )

        if elapsed > 0:
            print(f"Moves/second: {self.stats['total_moves']/elapsed:.2f}")
            print(f"Model calls/second: {self.stats['model_calls']/elapsed:.2f}")

        print(f"{'='*60}\n")


def get_tree_depth(node) -> int:
    """Calculate maximum depth of MCTS tree."""
    if not node or not node.expanded():
        return 0
    return 1 + max(
        (get_tree_depth(child) for child in node.children.values()), default=0
    )


def print_tree_info(mcts: MCTS, max_depth: int = 3):
    """Print information about the current MCTS tree."""
    if not mcts.root:
        print("No root node")
        return

    print(f"\nTree Info:")
    print(f"Root visits: {mcts.root.visit_count}")
    print(f"Root value: {mcts.root.value():.3f}")
    print(f"Children: {len(mcts.root.children)}")

    if mcts.root.children:
        print("Child action values (visits, avg_value, prior):")
        for action, child in sorted(mcts.root.children.items()):
            print(
                f"  Action {action}: {child.visit_count:3d} visits, "
                f"value={child.value():.3f}, prior={child.probability:.3f}"
            )


def run_single_game(
    worker_id: int = 0, verbose: bool = True, debug: bool = False
) -> List[MCTSStatistic]:
    """Run a single game with MCTS and return the game history."""

    # Set up queues
    requests = Queue()
    window = Queue()
    request_list = []
    window_list = []

    # Initialize MCTS
    mcts = MCTS(worker_id, requests, window)
    model = MockModel()
    monitor = MCTSMonitor()

    game_running = True
    step_count = 0

    if verbose:
        print(f"Starting game {worker_id}")
        print(f"Initial game state: {mcts.game}")

    while game_running:
        step_count += 1
        step_start = time.time()

        # Get model request from MCTS
        if len(request_list) > 0:
            worker, state = request_list.pop(0)
            policy, value = model.predict(state)
            monitor.stats["model_calls"] += 1

            if debug and step_count % 10 == 0:
                print(f"Step {step_count}: Model call for worker {worker}")
                print(f"  Policy: {policy}")
                print(f"  Value: {value:.3f}")
        else:
            # No model request yet, use dummy values
            policy = np.ones(6, dtype=np.float32) / 6
            value = np.float32(0.0)

        # Step MCTS
        request, result = mcts.step(policy, value)  # type: ignore
        if request is not None:
            request_list.append((0, request))
        elif result is not None:
            window_list.append(result)
            game_running = False  # Game ended, no more steps needed

        step_time = time.time() - step_start

        # Log move if this step resulted in an actual game move
        # if step_count % (CONFIG.num_simulations * 5) == 0:
        #     # monitor.log_move(
        #     #     step_time, mcts.root.visit_count, get_tree_depth(mcts.root) # type: ignore
        #     # )
        #     print(f"Move made? \n {mcts.game.get_state()}")

        if verbose and step_count % 80 == 0:
            print(
                f"Step {step_count}: Game still running, current player: {mcts.game.current_player}"
            )
            if debug:
                print_tree_info(mcts)

        # Safety check to prevent infinite loops
        if step_count > CONFIG.max_moves * CONFIG.num_simulations:
            print("WARNING: Stopping after 10000 steps to prevent infinite loop")
            break

    # Get completed game from window
    game_history = []
    if len(window_list) > 0:
        game_history = window_list.pop(0)
        print("Snatch!!!")
        monitor.log_game_complete(game_history)
    else:
        print("WARNING: No game history in window, something went wrong")

    if verbose:
        print(f"Game {worker_id} completed in {step_count} steps")
        print(f"Final game state: \n {mcts.game}")
        print(
            f"Winner: {mcts.game.winner if hasattr(mcts.game, 'winner') else 'Unknown'}"
        )
        print(f"Model calls made: {model.call_count}")
        monitor.print_stats()

    return game_history


def run_multiple_games(num_games: int = 5, verbose: bool = False):
    """Run multiple games and collect aggregate statistics."""

    print(f"Running {num_games} games...")
    start_time = time.time()

    all_histories = []
    total_model_calls = 0
    game_lengths = []

    for i in range(num_games):
        if verbose or (i + 1) % max(1, num_games // 10) == 0:
            print(f"Starting game {i+1}/{num_games}")

        history = run_single_game(worker_id=i, verbose=verbose, debug=False)
        all_histories.append(history)
        game_lengths.append(len(history))

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"MULTI-GAME RESULTS")
    print(f"{'='*60}")
    print(f"Total games: {num_games}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per game: {elapsed/num_games:.2f}s")
    print(f"Avg game length: {np.mean(game_lengths):.1f} moves")
    print(f"Games per minute: {num_games * 60 / elapsed:.1f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test MCTS implementation")
    parser.add_argument(
        "--single", action="store_true", help="Run single game with detailed output"
    )
    parser.add_argument(
        "--multi", type=int, default=5, help="Run multiple games (default: 5)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug output")
    parser.add_argument("--profile", action="store_true", help="Profile performance")

    args = parser.parse_args()

    if args.single:
        print("Running single game with detailed monitoring...")
        history = run_single_game(verbose=True, debug=args.debug)
        print(f"Game produced {len(history)} training examples")

    elif args.multi:
        run_multiple_games(args.multi, verbose=args.verbose)

    else:
        print("No mode specified. Use --single or --multi N")
        print("Examples:")
        print("  python test.py --single --debug")
        print("  python test.py --multi 10 --verbose")


if __name__ == "__main__":
    main()
