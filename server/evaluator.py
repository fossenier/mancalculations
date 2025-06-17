"""
Evaluation system for AlphaZero Kalah
Includes tournament play, benchmarking, and performance analysis
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import os

from kalah_game import KalahGame
from mcts import MCTS
from network import KalahNetwork
from config import AlphaZeroConfig


class RandomPlayer:
    """Random baseline player"""

    def __init__(self):
        self.name = "Random"

    def get_action(self, game: KalahGame) -> int:
        """Select random valid action"""
        valid_moves = game.get_valid_moves()
        valid_actions = np.where(valid_moves)[0]
        return np.random.choice(valid_actions)


class MinimaxPlayer:
    """Minimax player with configurable depth"""

    def __init__(self, depth: int = 5, use_endgame_db: bool = False):
        self.name = f"Minimax(depth={depth})"
        self.depth = depth
        self.use_endgame_db = use_endgame_db
        self.nodes_evaluated = 0

    def get_action(self, game: KalahGame) -> int:
        """Select action using minimax with alpha-beta pruning"""
        self.nodes_evaluated = 0
        _, action = self._minimax(game, self.depth, -float("inf"), float("inf"), True)
        return action

    def _minimax(
        self, game: KalahGame, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> Tuple[float, int]:
        """Minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1

        # Terminal node
        if game.game_over or depth == 0:
            return self._evaluate(game), -1

        valid_moves = game.get_valid_moves()
        valid_actions = np.where(valid_moves)[0]

        if maximizing:
            max_eval = -float("inf")
            best_action = valid_actions[0]

            for action in valid_actions:
                game_copy = game.clone()
                extra_turn = game_copy.make_move(action)

                # Handle extra turns
                if extra_turn and not game_copy.game_over:
                    eval_score, _ = self._minimax(game_copy, depth, alpha, beta, True)
                else:
                    eval_score, _ = self._minimax(
                        game_copy, depth - 1, alpha, beta, False
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
            best_action = valid_actions[0]

            for action in valid_actions:
                game_copy = game.clone()
                extra_turn = game_copy.make_move(action)

                # Handle extra turns
                if extra_turn and not game_copy.game_over:
                    eval_score, _ = self._minimax(game_copy, depth, alpha, beta, False)
                else:
                    eval_score, _ = self._minimax(
                        game_copy, depth - 1, alpha, beta, True
                    )

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval, best_action

    def _evaluate(self, game: KalahGame) -> float:
        """Evaluate game position"""
        if game.game_over:
            # Use actual game outcome
            return game.get_reward(0) * 100  # Scale for better distinction
        else:
            # Heuristic evaluation: stone difference
            return game.get_score_difference()


class AlphaZeroPlayer:
    """AlphaZero player using neural network and MCTS"""

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

        return action


class Evaluator:
    """System for evaluating model performance"""

    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.logger = logging.getLogger("Evaluator")

    def play_match(
        self, player1, player2, num_games: int = 100, verbose: bool = False
    ) -> Dict[str, any]:
        """Play a match between two players"""
        results = {
            "wins": [0, 0],
            "draws": 0,
            "total_score": [0, 0],
            "game_lengths": [],
            "score_differences": [],
        }

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
                current_player_idx = player_map[game.current_player]
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

            if game.winner == -1:
                results["draws"] += 1
            else:
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

    def evaluate_model(self, model_path: str, iteration: int) -> Dict[str, any]:
        """Comprehensive evaluation of a model"""
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
    ) -> Dict[str, any]:
        """Compare two models directly"""
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

    def _save_evaluation(self, evaluation_results: Dict[str, any]):
        """Save evaluation results to file"""
        filename = os.path.join(
            self.config.system.log_dir,
            f"evaluation_iter{evaluation_results['iteration']}.json",
        )

        with open(filename, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        self.logger.info(f"Saved evaluation results to {filename}")

    def analyze_game_quality(
        self, model_path: str, num_games: int = 10
    ) -> Dict[str, any]:
        """Analyze quality of games played by the model"""
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
