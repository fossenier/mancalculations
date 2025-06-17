"""
Distributed training system for AlphaZero Kalah on 4x A100 GPUs
Implements data parallel training with mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import pickle

from network import KalahNetwork
from config import AlphaZeroConfig


class ReplayBuffer(Dataset):
    """Experience replay buffer for self-play data"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def push(self, experiences: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add experiences to buffer (state, policy, value)"""
        for exp in experiences:
            if len(self.buffer) < self.max_size:
                self.buffer.append(exp)
            else:
                self.buffer[self.position] = exp
                self.position = (self.position + 1) % self.max_size

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, policy, value = self.buffer[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([value]),
        )

    def save(self, path: str):
        """Save buffer to disk"""
        with open(path, "wb") as f:
            pickle.dump({"buffer": self.buffer, "position": self.position}, f)

    def load(self, path: str):
        """Load buffer from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.buffer = data["buffer"]
            self.position = data["position"]


class AlphaZeroTrainer:
    """
    Distributed trainer for AlphaZero Kalah
    Manages multi-GPU training with optimal resource utilization
    """

    def __init__(self, config: AlphaZeroConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # Setup logging
        self._setup_logging()

        # Initialize distributed training
        self._init_distributed()

        # Create model
        self.model = KalahNetwork(config).to(self.device)
        self.model = DDP(self.model, device_ids=[rank])

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.training.training_steps_per_iteration
        )

        # Mixed precision training
        self.scaler = GradScaler()

        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")

        # Replay buffer (only on rank 0)
        if rank == 0:
            self.replay_buffer = ReplayBuffer(config.training.replay_buffer_size)
            self.writer = SummaryWriter(config.system.tensorboard_dir)

        # Training statistics
        self.iteration = 0
        self.total_steps = 0

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(
            self.config.system.log_dir,
            f'training_rank{self.rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        )

        logging.basicConfig(
            level=logging.INFO if self.config.system.verbose else logging.WARNING,
            format=f"[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _init_distributed(self):
        """Initialize distributed training"""
        os.environ["MASTER_ADDR"] = self.config.distributed.master_addr
        os.environ["MASTER_PORT"] = self.config.distributed.master_port

        dist.init_process_group(
            backend=self.config.distributed.backend,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Set device
        torch.cuda.set_device(self.rank)

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict:
        """Single training step"""
        states, target_policies, target_values = batch
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # Mixed precision forward pass
        with autocast():
            log_policies, values = self.model(states)

            # Calculate losses
            value_loss = self.value_loss_fn(values, target_values)
            policy_loss = self.policy_loss_fn(log_policies, target_policies)

            # Combined loss (as per AlphaZero paper)
            total_loss = value_loss + policy_loss

        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.gradient_clip_norm
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Return losses for logging
        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train_iteration(self, experiences: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Train for one iteration on new self-play data"""
        if self.rank == 0:
            # Add experiences to replay buffer
            self.replay_buffer.push(experiences)
            self.logger.info(f"Replay buffer size: {len(self.replay_buffer)}")

            # Wait for minimum buffer size
            if len(self.replay_buffer) < self.config.training.min_replay_size:
                self.logger.info("Waiting for more data...")
                return

        # Synchronize all processes
        dist.barrier()

        # Create data loader
        if self.rank == 0:
            dataset = self.replay_buffer
        else:
            # Other ranks create dummy dataset
            dataset = ReplayBuffer(1)
            dataset.push([(np.zeros(14), np.zeros(6), 0.0)])

        sampler = DistributedSampler(dataset, self.world_size, self.rank)
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size_per_gpu,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

        # Training loop
        self.model.train()
        epoch_stats = {"total_loss": 0, "value_loss": 0, "policy_loss": 0}

        for step in range(self.config.training.training_steps_per_iteration):
            for batch in loader:
                stats = self.train_step(batch)

                # Accumulate statistics
                for key, value in stats.items():
                    if key in epoch_stats:
                        epoch_stats[key] += value

                self.total_steps += 1

                # Log periodically
                if (
                    self.total_steps % self.config.system.log_interval == 0
                    and self.rank == 0
                ):
                    self._log_stats(stats)

        # Average statistics
        num_batches = len(loader) * self.config.training.training_steps_per_iteration
        for key in epoch_stats:
            epoch_stats[key] /= num_batches

        self.iteration += 1

        # Save checkpoint
        if self.iteration % self.config.training.checkpoint_interval == 0:
            self.save_checkpoint()

    def _log_stats(self, stats: dict):
        """Log training statistics"""
        self.logger.info(
            f"Iteration {self.iteration}, Step {self.total_steps}: "
            f"Loss={stats['total_loss']:.4f}, "
            f"Value={stats['value_loss']:.4f}, "
            f"Policy={stats['policy_loss']:.4f}, "
            f"LR={stats['lr']:.6f}"
        )

        # TensorBoard logging
        for key, value in stats.items():
            self.writer.add_scalar(f"training/{key}", value, self.total_steps)

    def save_checkpoint(self):
        """Save model checkpoint"""
        if self.rank != 0:
            return

        checkpoint = {
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config.to_dict(),
        }

        path = os.path.join(
            self.config.system.checkpoint_dir,
            f"{self.config.system.model_name}_iter{self.iteration}.pt",
        )
        torch.save(checkpoint, path)

        # Save replay buffer
        if self.config.system.save_replay_buffer:
            buffer_path = os.path.join(
                self.config.system.data_dir, f"replay_buffer_iter{self.iteration}.pkl"
            )
            self.replay_buffer.save(buffer_path)

        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.iteration = checkpoint["iteration"]
        self.total_steps = checkpoint["total_steps"]

        self.logger.info(f"Loaded checkpoint from iteration {self.iteration}")

    def get_current_model(self) -> KalahNetwork:
        """Get current model for evaluation"""
        return self.model.module

    def cleanup(self):
        """Cleanup distributed training"""
        if self.rank == 0:
            self.writer.close()
        dist.destroy_process_group()
