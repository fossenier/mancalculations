"""
Main training loop for AlphaZero Kalah
Coordinates self-play, training, and evaluation
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import os
import logging
import time
from datetime import datetime
import json
import argparse

from config import get_config, AlphaZeroConfig
from trainer import AlphaZeroTrainer
from self_play import SelfPlayManager
from evaluator import Evaluator
from typing import Optional


class AlphaZeroManager:
    """Main coordinator for AlphaZero training"""

    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.iteration = 0
        self.best_model_path = None

    def _setup_logging(self):
        """Setup main logger"""
        log_file = os.path.join(
            self.config.system.log_dir,
            f'alphazero_main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        return logging.getLogger("AlphaZeroManager")

    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        self.logger.info("Starting AlphaZero training")
        self.logger.info(
            f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}"
        )

        # Initialize components
        self_play_manager = SelfPlayManager(self.config)
        evaluator = Evaluator(self.config)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        # Main training loop
        start_time = time.time()

        while True:
            iteration_start = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting iteration {self.iteration}")
            self.logger.info(f"{'='*60}")

            # 1. Self-play phase
            self.logger.info("Phase 1: Self-play game generation")
            model_path = self._get_latest_model_path()

            experiences = self_play_manager.generate_games(model_path)
            self_play_manager.save_experiences(experiences, self.iteration)

            self_play_time = time.time() - iteration_start
            self.logger.info(f"Self-play completed in {self_play_time:.1f}s")

            # 2. Training phase
            self.logger.info("Phase 2: Neural network training")
            training_start = time.time()

            # Launch distributed training
            mp.spawn(
                self._train_worker,
                args=(self.config, experiences),
                nprocs=self.config.distributed.num_gpus,
                join=True,
            )

            training_time = time.time() - training_start
            self.logger.info(f"Training completed in {training_time:.1f}s")

            # 3. Evaluation phase
            if self.iteration % self.config.training.evaluation_interval == 0:
                self.logger.info("Phase 3: Model evaluation")
                eval_start = time.time()

                new_model_path = self._get_latest_model_path()

                # Evaluate new model
                eval_results = evaluator.evaluate_model(new_model_path, self.iteration)

                # Compare with previous best
                if self.best_model_path and self.best_model_path != new_model_path:
                    comparison = evaluator.compare_models(
                        new_model_path, self.best_model_path
                    )

                    # Update best model if new one is better
                    if comparison["model1_win_rate"] > 0.55:  # 55% win rate threshold
                        self.logger.info(
                            f"New best model! Win rate: {comparison['model1_win_rate']:.2%}"
                        )
                        self.best_model_path = new_model_path
                        self._save_best_model(new_model_path)
                else:
                    self.best_model_path = new_model_path

                eval_time = time.time() - eval_start
                self.logger.info(f"Evaluation completed in {eval_time:.1f}s")

            # Update iteration
            self.iteration += 1

            # Log iteration summary
            iteration_time = time.time() - iteration_start
            total_time = time.time() - start_time
            self.logger.info(
                f"\nIteration {self.iteration - 1} completed in {iteration_time:.1f}s"
            )
            self.logger.info(f"Total training time: {total_time / 3600:.1f} hours")

            # Save training state
            self._save_training_state()

            # Check for convergence (could add early stopping criteria here)
            if self._check_convergence(
                eval_results if "eval_results" in locals() else None
            ):
                self.logger.info("Training converged!")
                break

    def _train_worker(self, rank: int, config: AlphaZeroConfig, experiences):
        """Worker function for distributed training"""
        trainer = AlphaZeroTrainer(config, rank, config.distributed.num_gpus)

        try:
            # Load experiences if not rank 0
            if rank != 0:
                experiences = []  # Other ranks will use dummy data

            # Run training
            trainer.train_iteration(experiences)

        finally:
            trainer.cleanup()

    def _get_latest_model_path(self) -> Optional[str]:
        """Get path to latest model checkpoint"""
        if self.iteration == 0:
            return None

        checkpoint_dir = self.config.system.checkpoint_dir
        model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

        if not model_files:
            return None

        # Sort by iteration number
        model_files.sort(key=lambda x: int(x.split("iter")[1].split(".")[0]))
        latest_model = model_files[-1]

        return os.path.join(checkpoint_dir, latest_model)

    def _save_best_model(self, model_path: str):
        """Save the best model separately"""
        best_path = os.path.join(
            self.config.system.checkpoint_dir,
            f"{self.config.system.model_name}_best.pt",
        )

        # Copy the checkpoint
        import shutil

        shutil.copy2(model_path, best_path)

        self.logger.info(f"Saved best model to {best_path}")

    def _save_training_state(self):
        """Save current training state"""
        state = {
            "iteration": self.iteration,
            "best_model_path": self.best_model_path,
            "config": self.config.to_dict(),
        }

        state_path = os.path.join(
            self.config.system.checkpoint_dir, "training_state.json"
        )

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_checkpoint(self, checkpoint_path: str):
        """Resume from checkpoint"""
        # Load training state
        state_path = os.path.join(
            os.path.dirname(checkpoint_path), "training_state.json"
        )
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                self.iteration = state["iteration"]
                self.best_model_path = state.get("best_model_path")

        self.logger.info(f"Resumed from iteration {self.iteration}")

    def _check_convergence(self, eval_results: Optional[dict]) -> bool:
        """Check if training has converged"""
        # Simple convergence criteria - can be made more sophisticated
        if self.iteration >= 500:  # Max iterations
            return True

        if eval_results:
            # Check if performance against strong opponents has plateaued
            minimax7_win_rate = (
                eval_results["matches"].get("Minimax(depth=7)", {}).get("win_rate", 0)
            )
            if minimax7_win_rate > 0.95:  # Near-perfect play against Minimax-7
                return True

        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AlphaZero Kalah Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate", type=str, help="Path to model to evaluate")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Override number of GPUs if specified
    if args.num_gpus:
        config.distributed.num_gpus = args.num_gpus

    # Load custom config if provided
    if args.config:
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            # Update config with custom values
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    if args.evaluate:
        # Evaluation mode
        evaluator = Evaluator(config)
        results = evaluator.evaluate_model(args.evaluate, iteration=0)
        print(json.dumps(results, indent=2))
    else:
        # Training mode
        manager = AlphaZeroManager(config)
        manager.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
