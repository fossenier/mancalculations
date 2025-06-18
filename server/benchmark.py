"""
Time a single self-play game to estimate total runtime
"""

import time
import numpy as np
from self_play import SelfPlayWorker
from config import AlphaZeroConfig  # You'll need to import your config


def time_single_game(config, model_path=None):
    """
    Run exactly one self-play game and measure the time

    Args:
        config: Your AlphaZeroConfig instance
        model_path: Path to model (optional)

    Returns:
        tuple: (game_time_seconds, num_experiences, moves_played)
    """
    print("Initializing worker...")
    worker = SelfPlayWorker(worker_id=0, config=config, model_path=model_path)

    print("Starting single game...")
    start_time = time.time()

    experiences = worker.play_game()

    end_time = time.time()
    game_time = end_time - start_time

    num_experiences = len(experiences)
    moves_played = num_experiences  # Each experience = one move

    print(f"\n{'='*50}")
    print(f"SINGLE GAME TIMING RESULTS")
    print(f"{'='*50}")
    print(f"Game time: {game_time:.2f} seconds")
    print(f"Moves played: {moves_played}")
    print(f"Experiences generated: {num_experiences}")
    print(f"Time per move: {game_time/moves_played:.2f} seconds")
    print(f"{'='*50}")

    return game_time, num_experiences, moves_played


def estimate_total_runtime(config, model_path=None, num_samples=3):
    """
    Run multiple single games to get a better average estimate

    Args:
        config: Your AlphaZeroConfig instance
        model_path: Path to model (optional)
        num_samples: Number of games to average over

    Returns:
        dict: Runtime estimates
    """
    print(f"Running {num_samples} sample games for timing...")

    game_times = []
    total_experiences = []

    for i in range(num_samples):
        print(f"\nSample game {i+1}/{num_samples}")
        game_time, experiences, moves = time_single_game(config, model_path)
        game_times.append(game_time)
        total_experiences.append(experiences)

    avg_game_time = np.mean(game_times)
    std_game_time = np.std(game_times)
    avg_experiences = np.mean(total_experiences)

    # Get config values
    num_workers = config.self_play.num_workers
    games_per_worker = config.self_play.games_per_worker
    total_games = num_workers * games_per_worker

    # Estimate total runtime (accounting for parallelization)
    sequential_time = total_games * avg_game_time
    parallel_time = (
        games_per_worker * avg_game_time
    )  # Each worker runs games_per_worker games

    print(f"\n{'='*60}")
    print(f"RUNTIME ESTIMATION")
    print(f"{'='*60}")
    print(f"Average game time: {avg_game_time:.2f} ± {std_game_time:.2f} seconds")
    print(f"Average experiences per game: {avg_experiences:.1f}")
    print(f"")
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Games per worker: {games_per_worker}")
    print(f"  Total games: {total_games}")
    print(f"")
    print(f"Estimated runtimes:")
    print(f"  Sequential (1 worker): {sequential_time/60:.1f} minutes")
    print(f"  Parallel ({num_workers} workers): {parallel_time/60:.1f} minutes")
    print(f"  Total experiences expected: {int(total_games * avg_experiences)}")
    print(f"{'='*60}")

    return {
        "avg_game_time": avg_game_time,
        "std_game_time": std_game_time,
        "parallel_runtime_minutes": parallel_time / 60,
        "total_experiences": int(total_games * avg_experiences),
    }


if __name__ == "__main__":
    config = AlphaZeroConfig()
    time_single_game(config, None)
    # You'll need to load your config here
    # config = AlphaZeroConfig()  # or however you load it
    # model_path = "path/to/your/model.pth"  # optional

    # For quick single test:
    # time_single_game(config, model_path)

    # For better estimate with multiple samples:
    # estimate_total_runtime(config, model_path, num_samples=3)

    # print("Please uncomment and modify the config loading section above")
