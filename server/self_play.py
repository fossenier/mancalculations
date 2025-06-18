"""
Self-play game generation for AlphaZero Kalah
Runs parallel CPU processes to generate training data
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional
import logging
import time
from collections import deque
import pickle
import os
import random

from kalah_game import KalahGame
from mcts import MCTS
from network import KalahNetwork
from config import AlphaZeroConfig


class SelfPlayWorker:
    """Worker process for generating self-play games"""

    def __init__(
        self, worker_id: int, config: AlphaZeroConfig, model_path: Optional[str] = None
    ):
        self.worker_id = worker_id
        self.config = config
        self.model_path = model_path

        # Setup logging
        logging.basicConfig(
            level=logging.INFO if config.system.verbose else logging.WARNING,
            format=f"[Worker {worker_id}] %(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(f"SelfPlayWorker{worker_id}")

        # Load model
        self.network = self._load_model()
        self.mcts = MCTS(config, self.network)

    def _load_model(self) -> KalahNetwork:
        """Load neural network model"""
        network = KalahNetwork(self.config)

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="cpu")
            network.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Loaded model from {self.model_path}")
        else:
            self.logger.info("Using random initialized model")

        network.eval()
        return network

    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play a single self-play game
        Returns list of (state, policy, value) tuples
        """
        game = KalahGame()
        experiences = []
        move_count = 0

        # Temperature schedule
        temperature_threshold = self.config.mcts.temperature_threshold

        while not game.game_over and move_count < self.config.self_play.max_game_length:
            # Get canonical state
            canonical_state = game.get_canonical_state()

            # Run MCTS
            visits = self.mcts.search(game)

            # Calculate temperature
            if move_count < temperature_threshold:
                temperature = self.config.mcts.initial_temperature
            else:
                temperature = self.config.mcts.final_temperature

            # Get action probabilities
            valid_moves = game.get_valid_moves()
            if temperature == 0:
                # Greedy selection
                action_probs = np.zeros(6)
                action_probs[np.argmax(visits)] = 1.0
            else:
                # Apply temperature
                visits_temp = np.power(visits, 1.0 / temperature)
                action_probs = visits_temp / np.sum(visits_temp)

            # Store experience (will be labeled with game outcome later)
            experiences.append(
                (canonical_state.copy(), action_probs.copy(), game.current_player)
            )

            # Select action
            action = -1
            while action not in valid_moves:
                action = np.random.choice(6, replace=False, p=action_probs)

            # Make move
            extra_turn = game.make_move(action)
            move_count += 1

            # Clear MCTS tree periodically to save memory
            if move_count % 10 == 0:
                self.mcts.clear_tree()

        # Get game outcome
        if game.game_over:
            # Label experiences with actual game outcome
            labeled_experiences = []
            for state, policy, player in experiences:
                value = game.get_reward(player)
                labeled_experiences.append((state, policy, value))

            return labeled_experiences
        else:
            # Game hit max length, declare draw
            return [(state, policy, 0.0) for state, policy, _ in experiences]

    def run(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate multiple self-play games"""
        all_experiences = []

        for game_num in range(num_games):
            start_time = time.time()

            experiences = self.play_game()
            all_experiences.extend(experiences)

            game_time = time.time() - start_time

            if (game_num + 1) % 10 == 0:
                self.logger.info(
                    f"Games: {game_num + 1}/{num_games}, "
                    f"Experiences: {len(all_experiences)}, "
                    f"Time/game: {game_time:.2f}s"
                )

        return all_experiences


def run_self_play_worker(args):
    worker_id, config, model_path, num_games = args
    worker = SelfPlayWorker(worker_id, config, model_path)
    return worker.run(num_games)


class SelfPlayManager:
    """Manages parallel self-play game generation"""

    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.logger = logging.getLogger("SelfPlayManager")

    def generate_games(
        self, model_path: Optional[str] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate self-play games using multiple CPU workers"""
        num_workers = self.config.self_play.num_workers
        games_per_worker = self.config.self_play.games_per_worker

        self.logger.info(
            f"Starting {num_workers} workers to generate {num_workers * games_per_worker} games"
        )

        # Prepare argument tuples for each worker
        worker_args = [
            (worker_id, self.config, model_path, games_per_worker)
            for worker_id in range(num_workers)
        ]

        # Use multiprocessing pool to parallelize work
        with mp.Pool(num_workers) as pool:
            self.logger.info(
                "Starting self-play workers..."
            )  # This only runs once for 64 threads
            results = pool.map(run_self_play_worker, worker_args)

        # Flatten all experiences
        all_experiences = [exp for worker_result in results for exp in worker_result]

        self.logger.info(f"Total experiences generated: {len(all_experiences)}")

        random.shuffle(all_experiences)

        return all_experiences

    def save_experiences(
        self, experiences: List[Tuple[np.ndarray, np.ndarray, float]], iteration: int
    ):
        """Save experiences to disk"""
        filename = os.path.join(
            self.config.system.data_dir, f"experiences_iter{iteration}.pkl"
        )

        with open(filename, "wb") as f:
            pickle.dump(experiences, f)

        self.logger.info(f"Saved {len(experiences)} experiences to {filename}")

    def load_experiences(
        self, iteration: int
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Load experiences from disk"""
        filename = os.path.join(
            self.config.system.data_dir, f"experiences_iter{iteration}.pkl"
        )

        with open(filename, "rb") as f:
            experiences = pickle.load(f)

        self.logger.info(f"Loaded {len(experiences)} experiences from {filename}")
        return experiences


# Import torch here to avoid issues with multiprocessing
import torch
