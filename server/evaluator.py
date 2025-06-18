"""
Evaluation system for AlphaZero Kalah
Includes tournament play, benchmarking, and performance analysis
"""

import numpy as np
import numpy.typing as npt
import torch
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import os

from kalah_game import KalahGame
from mcts import MCTS
from network import KalahNetwork
from config import AlphaZeroConfig


class RandomPlayer:
    """
    Player making random moves as a baseline
    """

    def __init__(self) -> None:
        """
        Initializes the instance with a default name attribute set to 'Random'.
        """
        self.name = "Random"

    def get_action(self, game: KalahGame) -> int:
        """
        Selects and returns a random valid action for the given KalahGame instance.

        Args:
            game (KalahGame): The current game state from which to select a valid action.

        Returns:
            int: The index of the randomly selected valid action.
        """
        valid_moves = game.get_valid_moves()
        valid_actions = np.where(valid_moves)[0]
        return np.random.choice(valid_actions)


class MinimaxPlayer:
    """
    Minimax player with configurable depth
    """

    def __init__(self, depth: int = 5, use_endgame_db: bool = False):
        self.name = f"Minimax(depth={depth})"
        self.depth = depth
        self.use_endgame_db = use_endgame_db
        self.nodes_evaluated = 0

    def get_action(self, game: KalahGame) -> int:
        """
        Determines the best action to take for the given game state using the minimax algorithm with alpha-beta pruning.

        Args:
            game (KalahGame): The current state of the Kalah game.

        Returns:
            int: The index of the selected action (move) based on the minimax evaluation.
        """
        self.nodes_evaluated = 0
        _, action = self._minimax(game, self.depth, -float("inf"), float("inf"), True)
        return action

    def _minimax(
        self, game: KalahGame, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> Tuple[float, int]:
        """
        Implements the Minimax algorithm with alpha-beta pruning for the Kalah game.
        This recursive function evaluates the best possible move for the current player
        by simulating all possible moves up to a given depth, alternating between maximizing
        and minimizing players. Alpha-beta pruning is used to eliminate branches that cannot
        possibly affect the final decision, improving efficiency.
        Args:
            game (KalahGame): The current state of the Kalah game.
            depth (int): The maximum depth to search in the game tree.
            alpha (float): The best value that the maximizing player can guarantee so far.
            beta (float): The best value that the minimizing player can guarantee so far.
            maximizing (bool): True if the current player is the maximizing player, False otherwise.
        Returns:
            Tuple[float, int]: A tuple containing the evaluation score and the best action (move index).
                The score is positive if favorable for the maximizing player, negative otherwise.
                The action is the index of the best move, or -1 if at a terminal node.
        """

        self.nodes_evaluated += 1

        # Terminal node
        if game.game_over or depth == 0:
            evaluation = self._evaluate(game)
            # Positive values indicate a favourable evaluation, must negate for
            # minimizing player
            if maximizing:
                return evaluation, -1
            else:
                return -evaluation, -1

        valid_moves = game.get_valid_moves()
        # Extracts array of indices where valid_moves is True (AKA the valid actions [0-5])
        valid_actions: npt.NDArray[np.intp] = np.where(valid_moves)[0]

        if maximizing:
            max_eval = -float("inf")
            best_action: int = valid_actions[0]

            for action in valid_actions:
                action = int(action)
                game_copy = game.clone()
                extra_turn = game_copy.make_move(action)

                # Same player goes again if they land in their own store, AND ALSO at
                # the end of the game, so that scoring happens right
                if extra_turn or game_copy.game_over:
                    eval_score, _ = self._minimax(
                        game_copy, depth - 1, alpha, beta, maximizing
                    )
                else:
                    eval_score, _ = self._minimax(
                        game_copy, depth - 1, alpha, beta, not maximizing
                    )

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval, best_action
        else:
            min_eval = float("inf")
            best_action: int = valid_actions[0]

            for action in valid_actions:
                action = int(action)
                game_copy = game.clone()
                extra_turn = game_copy.make_move(action)

                # Same player goes again if they land in their own store, AND ALSO at
                # the end of the game, so that scoring happens right
                if extra_turn or game_copy.game_over:
                    eval_score, _ = self._minimax(
                        game_copy, depth, alpha, beta, maximizing
                    )
                else:
                    eval_score, _ = self._minimax(
                        game_copy, depth - 1, alpha, beta, not maximizing
                    )

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval, best_action

    def _evaluate(self, game: KalahGame) -> float:
        """
        Evaluates the current state of the Kalah game from the perspective of the active player.

        If the game is over, returns a boosted reward value for a true victory.
        Otherwise, returns a heuristic evaluation based on the difference in stones.

        Args:
            game (KalahGame): The current game state to evaluate.

        Returns:
            float: The evaluation score for the active player's position.
        """
        # The better the active player's position, the higher the evaluation
        if game.game_over:
            # Return a boosted value for true victory
            return game.get_reward() * 100
        else:
            # Heuristic evaluation: stone difference
            return game.get_score_difference()


class AlphaZeroPlayer:
    """
    AlphaZero player using neural network and MCTS
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        model_path: str,
        num_simulations: Optional[int] = None,
    ):
        self.name = f"AlphaZero(sims={num_simulations or config.mcts.num_simulations})"
        self.config = config

        # Override simulation count if specified
        if num_simulations:
            config.mcts.num_simulations = num_simulations

        # Load model
        self.network = KalahNetwork(config)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()

        self.mcts = MCTS(config, self.network)

    def get_action(self, game: KalahGame) -> int:
        """Select action using MCTS"""
        visits = self.mcts.search(game)

        # Greedy selection for evaluation
        action = np.argmax(visits)

        # Clear tree to save memory
        self.mcts.clear_tree()

        return int(action)


class Evaluator:
    """
    System for evaluating model performance
    """

    def __init__(self, config: AlphaZeroConfig) -> None:
        """
        Initializes the Evaluator with the given AlphaZero configuration.
        Args:
            config (AlphaZeroConfig): The configuration object for AlphaZero.
        Attributes:
            config (AlphaZeroConfig): Stores the provided configuration.
            logger (logging.Logger): Logger instance for the Evaluator class.
        """

        self.config = config
        self.logger = logging.getLogger("Evaluator")

    def play_match(
        self,
        player1: RandomPlayer | MinimaxPlayer | AlphaZeroPlayer,
        player2: RandomPlayer | MinimaxPlayer | AlphaZeroPlayer,
        num_games: int = 100,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        results = {
            "wins": [0, 0],
            "draws": 0,
            "total_score": [0, 0],
            "game_lengths": [],
            "score_differences": [],
        }
        """
        Simulates a series of Kalah matches between two players and collects statistics.

        Args:
            player1 (RandomPlayer | MinimaxPlayer | AlphaZeroPlayer): The first player instance.
            player2 (RandomPlayer | MinimaxPlayer | AlphaZeroPlayer): The second player instance.
            num_games (int, optional): Number of games to play. Defaults to 100.
            verbose (bool, optional): If True, renders the first game step-by-step. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary containing match statistics:
                - "wins": List[int] - Number of wins for each player.
                - "draws": int - Number of drawn games.
                - "total_score": List[int] - Total score accumulated by each player.
                - "game_lengths": List[int] - Number of moves in each game.
                - "score_differences": List[int] - Score difference for each game.
                - "win_rate": List[float] - Win rate for each player.
                - "draw_rate": float - Draw rate.
                - "avg_score": List[float] - Average score for each player.
                - "avg_game_length": float - Average number of moves per game.
                - "avg_score_diff": float - Average score difference per game.
        """

        for game_num in range(num_games):
            # Alternate starting player
            if game_num % 2 == 0:
                players = [player1, player2]
                player_map = [0, 1]
            else:
                players = [player2, player1]
                player_map = [1, 0]

            game = KalahGame()
            move_count = 0

            while (
                not game.game_over
                and move_count < self.config.self_play.max_game_length
            ):
                current_player = players[game.current_player]

                # Get action
                action = current_player.get_action(game)

                # Make move
                game.make_move(action)
                move_count += 1

                if verbose and game_num == 0:
                    game.render()

            # Record results
            results["game_lengths"].append(move_count)
            results["score_differences"].append(game.get_score_difference())

            # The game ended in a draw
            if game.winner == -1:
                results["draws"] += 1
            # The game had a winner
            elif game.winner is not None:
                winner_idx = player_map[game.winner]
                results["wins"][winner_idx] += 1

            results["total_score"][0] += (
                game.board[6] if player_map[0] == 0 else game.board[13]
            )
            results["total_score"][1] += (
                game.board[13] if player_map[1] == 1 else game.board[6]
            )

            if (game_num + 1) % 10 == 0:
                self.logger.info(f"Completed {game_num + 1}/{num_games} games")

        # Calculate statistics
        results["win_rate"] = [w / num_games for w in results["wins"]]
        results["draw_rate"] = results["draws"] / num_games
        results["avg_score"] = [s / num_games for s in results["total_score"]]
        results["avg_game_length"] = np.mean(results["game_lengths"])
        results["avg_score_diff"] = np.mean(results["score_differences"])

        return results

    def evaluate_model(self, model_path: str, iteration: int) -> Dict[str, Any]:
        """
        Evaluates a trained AlphaZero model by playing matches against various
        opponents and performing self-play with different MCTS simulation counts.

        Args:
            model_path (str): Path to the trained model to evaluate.
            iteration (int): The training iteration number corresponding to the model.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results, including:
                - 'iteration': The iteration number.
                - 'timestamp': ISO-formatted timestamp of evaluation.
                - 'model_path': Path to the evaluated model.
                - 'matches': Results against each opponent, keyed by opponent name,
                with win rates, draw rates, average scores, and other statistics.
                - 'mcts_scaling': Results of self-play between models with different
                MCTS simulation counts.

        Side Effects:
            - Logs evaluation progress and results.
            - Saves evaluation results to persistent storage via `_save_evaluation`.
        """
        self.logger.info(f"Evaluating model from iteration {iteration}")

        evaluation_results = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "matches": {},
        }

        # Create AlphaZero player
        az_player = AlphaZeroPlayer(self.config, model_path)

        # Test against different opponents
        opponents = [
            RandomPlayer(),
            MinimaxPlayer(depth=3),
            MinimaxPlayer(depth=5),
            MinimaxPlayer(depth=7),
        ]

        for opponent in opponents:
            self.logger.info(f"Playing against {opponent.name}")

            results = self.play_match(
                az_player, opponent, num_games=self.config.training.evaluation_games
            )

            evaluation_results["matches"][opponent.name] = {
                "win_rate": results["win_rate"][0],
                "opponent_win_rate": results["win_rate"][1],
                "draw_rate": results["draw_rate"],
                "avg_score": results["avg_score"][0],
                "opponent_avg_score": results["avg_score"][1],
                "avg_game_length": results["avg_game_length"],
                "avg_score_diff": results["avg_score_diff"],
            }

            self.logger.info(
                f"vs {opponent.name}: "
                f"Win rate: {results['win_rate'][0]:.2%}, "
                f"Avg score: {results['avg_score'][0]:.1f} - {results['avg_score'][1]:.1f}"
            )

        # Self-play with different MCTS simulations
        self.logger.info("Testing MCTS scaling")
        base_player = AlphaZeroPlayer(self.config, model_path, num_simulations=400)
        strong_player = AlphaZeroPlayer(self.config, model_path, num_simulations=1600)

        scaling_results = self.play_match(strong_player, base_player, num_games=50)
        evaluation_results["mcts_scaling"] = {
            "strong_vs_weak_win_rate": scaling_results["win_rate"][0],
            "simulation_ratio": 1600 / 400,
        }

        # Save evaluation results
        self._save_evaluation(evaluation_results)

        return evaluation_results

    def compare_models(
        self, model1_path: str, model2_path: str, num_games: int = 100
    ) -> Dict[str, Any]:
        """
        Compares two AlphaZero models by playing a specified number of games between them.

        Args:
            model1_path (str): Path to the first model's checkpoint or weights file.
            model2_path (str): Path to the second model's checkpoint or weights file.
            num_games (int, optional): Number of games to play for the comparison. Defaults to 100.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "model1": Path to the first model.
                - "model2": Path to the second model.
                - "model1_win_rate": Win rate of the first model.
                - "model2_win_rate": Win rate of the second model.
                - "draw_rate": Rate of draws between the two models.
                - "avg_game_length": Average length of the games played.
                - "timestamp": ISO formatted timestamp of when the comparison was performed.
        """
        player1 = AlphaZeroPlayer(self.config, model1_path)
        player2 = AlphaZeroPlayer(self.config, model2_path)

        self.logger.info(f"Comparing models: {model1_path} vs {model2_path}")

        results = self.play_match(player1, player2, num_games)

        comparison = {
            "model1": model1_path,
            "model2": model2_path,
            "model1_win_rate": results["win_rate"][0],
            "model2_win_rate": results["win_rate"][1],
            "draw_rate": results["draw_rate"],
            "avg_game_length": results["avg_game_length"],
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"Model comparison: "
            f"Model1 win rate: {results['win_rate'][0]:.2%}, "
            f"Model2 win rate: {results['win_rate'][1]:.2%}"
        )

        return comparison

    def _save_evaluation(self, evaluation_results: Dict[str, Any]):
        """
        Save evaluation results to a JSON file.

        Args:
            evaluation_results (Dict[str, Any]): A dictionary containing the evaluation results,
                including an 'iteration' key used to name the output file.

        Side Effects:
            Writes the evaluation results to a JSON file in the log directory specified by the configuration.
            Logs the file path where the results are saved.
        """
        filename = os.path.join(
            self.config.system.log_dir,
            f"evaluation_iter{evaluation_results['iteration']}.json",
        )

        with open(filename, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        self.logger.info(f"Saved evaluation results to {filename}")

    def analyze_game_quality(
        self, model_path: str, num_games: int = 10
    ) -> Dict[str, Any]:
        """
        Analyzes the quality of games played by an AlphaZero-based player.

        Simulates a number of games using the provided model and collects various statistics
        to evaluate the quality and characteristics of the gameplay. The analysis includes
        move distributions, game lengths, score differences, capture frequencies, and extra turn
        frequencies. Summary statistics such as average game length, standard deviation of game
        length, average absolute score difference, average capture frequency, average extra turn
        frequency, and move entropy are computed.

        Args:
            model_path (str): Path to the trained AlphaZero model to be evaluated.
            num_games (int, optional): Number of games to simulate for analysis. Defaults to 10.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - 'avg_game_length': Average length of the simulated games.
                - 'std_game_length': Standard deviation of game lengths.
                - 'avg_score_diff': Average absolute score difference between players.
                - 'avg_capture_freq': Average frequency of capture events per move.
                - 'avg_extra_turn_freq': Average frequency of extra turns per move.
                - 'move_entropy': Entropy of the move distribution across all games.
        """
        player = AlphaZeroPlayer(self.config, model_path)

        analysis = {
            "move_distributions": [],
            "game_lengths": [],
            "score_differences": [],
            "capture_frequencies": [],
            "extra_turn_frequencies": [],
        }

        for _ in range(num_games):
            game = KalahGame()
            move_count = 0
            captures = 0
            extra_turns = 0
            move_dist = np.zeros(6)

            while (
                not game.game_over
                and move_count < self.config.self_play.max_game_length
            ):
                # Track move distribution
                action = player.get_action(game)
                move_dist[action] += 1

                # Track game events
                old_scores = [game.board[6], game.board[13]]
                extra_turn = game.make_move(action)
                new_scores = [game.board[6], game.board[13]]

                if extra_turn:
                    extra_turns += 1

                # Check for captures
                if (
                    new_scores[game.current_player] - old_scores[game.current_player]
                    > game.board[action]
                ):
                    captures += 1

                move_count += 1

            analysis["move_distributions"].append(move_dist / move_count)
            analysis["game_lengths"].append(move_count)
            analysis["score_differences"].append(game.get_score_difference())
            analysis["capture_frequencies"].append(captures / move_count)
            analysis["extra_turn_frequencies"].append(extra_turns / move_count)

        # Calculate statistics
        return {
            "avg_game_length": np.mean(analysis["game_lengths"]),
            "std_game_length": np.std(analysis["game_lengths"]),
            "avg_score_diff": np.mean(np.abs(analysis["score_differences"])),
            "avg_capture_freq": np.mean(analysis["capture_frequencies"]),
            "avg_extra_turn_freq": np.mean(analysis["extra_turn_frequencies"]),
            "move_entropy": -np.sum(
                np.mean(analysis["move_distributions"], axis=0)
                * np.log(np.mean(analysis["move_distributions"], axis=0) + 1e-8)
            ),
        }
