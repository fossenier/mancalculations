"""
Configuration file for AlphaZero Kalah training on 4x A100 GPUs
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""

    input_dim: int = 12  # 12 pits
    embedding_dim: int = 128
    num_residual_blocks: int = 20  # 15-25 recommended
    num_filters: int = 256  # 256-512 recommended
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    value_head_hidden_dim: int = 128
    policy_output_dim: int = 6  # 6 possible moves in Kalah


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search configuration"""

    num_simulations: int = 1200  # 800-1600 recommended
    c_puct: float = 2.0  # 1.5-2.5 for Kalah's tactical nature
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 30  # Moves before switching to greedy
    initial_temperature: float = 1.0
    final_temperature: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    batch_size_per_gpu: int = 2048
    total_batch_size: int = 8192  # 4 GPUs * 2048
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    replay_buffer_size: int = 1_000_000
    min_replay_size: int = 10_000
    games_per_iteration: int = 30_000  # 25k-50k recommended
    training_steps_per_iteration: int = 1000
    checkpoint_interval: int = 10  # Save every N iterations
    evaluation_interval: int = 5
    evaluation_games: int = 100


@dataclass
class SelfPlayConfig:
    """Self-play game generation configuration"""

    num_workers: int = 64  # CPU processes for game generation
    games_per_worker: int = 2
    max_game_length: int = 200  # Maximum moves before draw


@dataclass
class DistributedConfig:
    """Multi-GPU training configuration"""

    num_gpus: int = 4
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = False
    use_mixed_precision: bool = True
    fp16_opt_level: str = "O2"  # Apex optimization level


@dataclass
class SystemConfig:
    """System and I/O configuration"""

    data_dir: str = "./kalah_data"
    checkpoint_dir: str = "./kalah_checkpoints"
    log_dir: str = "./kalah_logs"
    tensorboard_dir: str = "./kalah_tensorboard"
    model_name: str = "kalah_alphazero"
    save_replay_buffer: bool = True
    log_interval: int = 100
    verbose: bool = True


@dataclass
class AlphaZeroConfig:
    """Master configuration combining all components"""

    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [
            self.system.data_dir,
            self.system.checkpoint_dir,
            self.system.log_dir,
            self.system.tensorboard_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {
            "model": self.model.__dict__,
            "mcts": self.mcts.__dict__,
            "training": self.training.__dict__,
            "self_play": self.self_play.__dict__,
            "distributed": self.distributed.__dict__,
            "system": self.system.__dict__,
        }


def get_config():
    """Get default configuration"""
    return AlphaZeroConfig()
