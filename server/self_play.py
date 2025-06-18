"""
Self-play game generation for AlphaZero Kalah
Runs parallel CPU processes to generate training data
"""

import numpy as np
import numpy.typing as npt
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
        self,
        worker_id: int,
        config: AlphaZeroConfig,
        model_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> None:
        """
        Initializes a SelfPlayWorker instance with GPU support.

        Args:
            worker_id (int): Unique identifier for the worker.
            config (AlphaZeroConfig): Configuration object containing parameters for AlphaZero.
            model_path (Optional[str], optional): Path to the pre-trained model file. Defaults to None.
            gpu_id (Optional[int], optional): GPU device ID to use. If None, uses CPU.
        """
        self.worker_id = worker_id
        self.config = config
        self.model_path = model_path
        self.gpu_id = gpu_id

        # Set device
        if gpu_id is not None:
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cpu"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO if config.system.verbose else logging.WARNING,
            format=f"[Worker {worker_id}] %(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(f"SelfPlayWorker{worker_id}")

        # Load model on specific device
        self.network = self._load_model()

        # Create MCTS with larger batch size for GPU
        batch_size = 64 if self.device.startswith("cuda") else 32
        self.mcts = MCTS(config, self.network, batch_size=batch_size)

        self.logger.info(f"Worker {worker_id} initialized on {self.device}")

    def _load_model(self) -> KalahNetwork:
        """
        Loads a KalahNetwork model optimized for GPU inference
        """
        network = KalahNetwork(self.config, device=self.device)

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            network.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Loaded model from {self.model_path}")
        else:
            self.logger.info("Using random initialized model")

        # Already in eval mode and on correct device from __init__
        # Optionally compile for faster inference
        if self.device.startswith("cuda"):
            compiled_network = network.compile_for_inference()
            # If compilation was successful, use the compiled version
            if isinstance(compiled_network, torch.jit.ScriptModule):
                return KalahNetwork(compiled_network)

        return network

    def play_game(
        self,
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
        """
        Plays a single self-play game of Kalah using Monte Carlo Tree Search (MCTS) and returns the collected experiences.

        The method simulates a game between two agents, using MCTS to select moves according to a temperature schedule.
        Each move's state, action probabilities, and player are recorded as experiences. At the end of the game, each
        experience is labeled with the final game outcome from the perspective of the player who made the move.

        Returns:
            List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
                A list of tuples, each containing:
                    - The canonical state (numpy array) at the time of the move,
                    - The action probability distribution (numpy array) used to select the move,
                    - The game outcome (float) from the perspective of the player who made the move.
                If the game ends normally, the outcome is the actual reward; if the game hits the maximum length, the outcome is 0.0 (draw).
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
                # Mask invalid moves by setting their visit counts to -inf
                masked_visits = np.copy(visits)
                masked_visits[np.logical_not(valid_moves)] = -np.inf
                action_probs[np.argmax(masked_visits)] = 1.0
            else:
                # Apply temperature, mask invalid moves
                visits_temp = np.zeros(6)
                visits_temp[valid_moves] = np.power(
                    visits[valid_moves], 1.0 / temperature
                )
                if np.sum(visits_temp) > 0:
                    action_probs = visits_temp / np.sum(visits_temp)
                else:
                    # If all visits are zero (shouldn't happen), pick uniform over valid moves
                    action_probs = np.zeros(6)
                    action_probs[valid_moves] = 1.0 / np.sum(valid_moves)

            # Store experience (will be labeled with game outcome later)
            experiences.append(
                (canonical_state.copy(), action_probs.copy(), game.current_player)
            )

            # Select action
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(6, p=action_probs)

            # Add this check:
            if not valid_moves[action]:
                print(f"ERROR: Selected invalid action {action}")
                print(f"Valid moves: {valid_moves}")
                print(f"Action probs: {action_probs}")
                # Force select from valid moves
                action = np.random.choice(np.where(valid_moves)[0])

            # Make move
            extra_turn = game.make_move(action)

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

    def run(
        self, num_games: int
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
        """
        Runs a specified number of self-play games, collecting and returning experiences from each game.
        Args:
            num_games (int): The number of games to play.
        Returns:
            List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
                A list of experiences collected from all games. Each experience is a tuple containing:
                    - State (npt.NDArray[np.float64])
                    - Action (npt.NDArray[np.float64])
                    - Reward (float)
        Logs progress every 10 games, including the number of games played, total experiences collected, and time taken per game.
        """

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


def run_self_play_worker(
    args: tuple,
) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
    """
    Runs a self-play worker to generate game data.

    Args:
        args (tuple): A tuple containing the following elements:
            - worker_id (int): The unique identifier for the worker.
            - config (Any): Configuration object for the self-play worker.
            - model_path (str): Path to the model to be used for self-play.
            - num_games (int): Number of games to run in self-play.
            - gpu_id (Optional[int]): GPU device ID to use (if applicable).

    Returns:
        List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
            A list of tuples, each containing:
                - The state array (np.ndarray of float64)
                - The policy array (np.ndarray of float64)
                - The game result (float)
    """
    worker_id, config, model_path, num_games, gpu_id = args
    worker = SelfPlayWorker(worker_id, config, model_path, gpu_id=gpu_id)
    return worker.run(num_games)


class SelfPlayManager:
    """Manages parallel self-play game generation"""

    def __init__(self, config: AlphaZeroConfig):
        """
        Initializes the SelfPlayManager with the given AlphaZero configuration.

        Args:
            config (AlphaZeroConfig): The configuration object containing parameters for AlphaZero.

        Attributes:
            config (AlphaZeroConfig): Stores the provided AlphaZero configuration.
            logger (logging.Logger): Logger instance for the SelfPlayManager class.
        """
        self.config = config
        self.logger = logging.getLogger("SelfPlayManager")

    def generate_games(
        self, model_path: Optional[str] = None, num_gpus: int = 4
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
        """
        Generates self-play game experiences in parallel using multiple GPUs.

        Args:
            model_path (Optional[str], optional): Path to the model to use for self-play. If None, uses the default model. Defaults to None.
            num_gpus (int, optional): Number of GPUs available for distributing workers. Defaults to 4.

        Returns:
            List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
                A list of tuples, each containing the state array, policy array, and reward for each experience generated during self-play.

        Logs:
            - Number of workers and games to be generated.
            - Distribution of workers across GPUs.
            - Total number of experiences generated.
        """
        num_workers = self.config.self_play.num_workers
        games_per_worker = self.config.self_play.games_per_worker

        self.logger.info(
            f"Starting {num_workers} workers to generate {num_workers * games_per_worker} games"
        )

        # Distribute workers across available GPUs
        worker_args = []
        for worker_id in range(num_workers):
            gpu_id = worker_id % num_gpus if num_gpus > 0 else None
            worker_args.append(
                (worker_id, self.config, model_path, games_per_worker, gpu_id)
            )

        # Use multiprocessing pool
        with mp.Pool(num_workers) as pool:
            self.logger.info(f"Starting self-play workers across {num_gpus} GPUs...")
            results = pool.map(run_self_play_worker, worker_args)

        # Flatten all experiences
        all_experiences = [exp for worker_result in results for exp in worker_result]

        self.logger.info(f"Total experiences generated: {len(all_experiences)}")

        random.shuffle(all_experiences)

        return all_experiences

    def save_experiences(
        self,
        experiences: List[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]
        ],
        iteration: int,
    ) -> None:
        """
        Saves a list of experience tuples to disk as a pickle file.

        Each experience is a tuple containing two NumPy arrays (state and action representations)
        and a float (reward). The experiences are saved to a file named according to the given
        iteration number in the configured data directory.

        Args:
            experiences (List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]):
                A list of experience tuples to be saved.
            iteration (int): The current iteration number, used to name the output file.

        Returns:
            None
        """
        filename = os.path.join(
            self.config.system.data_dir, f"experiences_iter{iteration}.pkl"
        )

        with open(filename, "wb") as f:
            pickle.dump(experiences, f)

        self.logger.info(f"Saved {len(experiences)} experiences to {filename}")

    def load_experiences(
        self, iteration: int
    ) -> List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
        """
        Loads experience tuples from a pickle file for a given training iteration.

        Args:
            iteration (int): The iteration number whose experiences should be loaded.

        Returns:
            List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]:
                A list of experience tuples, where each tuple contains:
                    - state (np.ndarray): The state representation as a NumPy array of floats.
                    - action (np.ndarray): The action representation as a NumPy array of floats.
                    - reward (float): The reward value associated with the experience.

        Logs:
            The number of loaded experiences and the filename they were loaded from.
        """
        filename = os.path.join(
            self.config.system.data_dir, f"experiences_iter{iteration}.pkl"
        )

        with open(filename, "rb") as f:
            experiences = pickle.load(f)

        self.logger.info(f"Loaded {len(experiences)} experiences from {filename}")
        return experiences


# Import torch here to avoid issues with multiprocessing
import torch
