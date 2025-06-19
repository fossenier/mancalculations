"""
Neural network architecture for Kalah AlphaZero
Optimized for GPU batch inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


class ResidualBlock1D(nn.Module):
    """1D Residual block with batch normalization"""

    def __init__(self, num_filters: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(num_filters)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class KalahNetwork(nn.Module):
    """
    AlphaZero network for Kalah
    Optimized for GPU batch inference
    """

    def __init__(self, config, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Input embedding layer
        self.embedding = nn.Linear(config.model.input_dim, config.model.embedding_dim)

        # Initial convolution to transform to CNN format
        self.initial_conv = nn.Conv1d(
            1, config.model.num_filters, kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm1d(config.model.num_filters)

        # Residual blocks with multiple kernel sizes
        self.residual_blocks = nn.ModuleList()
        for i in range(config.model.num_residual_blocks):
            kernel_size = config.model.kernel_sizes[i % len(config.model.kernel_sizes)]
            self.residual_blocks.append(
                ResidualBlock1D(config.model.num_filters, kernel_size)
            )

        # Global feature extraction
        self.global_pool_channels = config.model.num_filters * 2  # Max + Avg pool

        # Dense layers
        self.fc1 = nn.Linear(self.global_pool_channels, 256)
        self.fc2 = nn.Linear(256, 128)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, config.model.policy_output_dim), nn.LogSoftmax(dim=1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, config.model.value_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.model.value_head_hidden_dim, 1),
            nn.Tanh(),
        )

        # Initialize weights
        self._initialize_weights()

        # Move to device and set to eval mode
        self.to(self.device)
        self.eval()

        # Disable gradient computation for inference
        for param in self.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Board state tensor [batch_size, 14]
        Returns:
            log_policy: Log probabilities for moves [batch_size, 6]
            value: Position evaluation [-1, 1] [batch_size, 1]
        """
        batch_size = x.size(0)

        # Embedding layer
        x = self.embedding(x)  # [batch, embedding_dim]
        x = F.relu(x)

        # Reshape for 1D convolution
        x = x.unsqueeze(1)  # [batch, 1, embedding_dim]

        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Global pooling (max + average)
        max_pool = F.max_pool1d(x, kernel_size=x.size(2))
        avg_pool = F.avg_pool1d(x, kernel_size=x.size(2))
        x = torch.cat([max_pool, avg_pool], dim=1)

        # Flatten
        x = x.view(batch_size, -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Dual heads
        log_policy = self.policy_head(x)
        value = self.value_head(x)

        return log_policy, value

    @torch.no_grad()
    def predict(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single board state
        Optimized to avoid repeated mode switching
        Args:
            board_state: Board state array [14]
        Returns:
            policy: Move probabilities [6]
            value: Position evaluation scalar
        """
        # Convert to tensor and add batch dimension
        x = torch.FloatTensor(board_state).unsqueeze(0).to(self.device)

        log_policy, value = self(x)

        # Convert to numpy
        policy = torch.exp(log_policy).cpu().numpy()[0]
        value = value.cpu().numpy()[0, 0]

        return policy, value

    @torch.no_grad()
    def predict_batch(
        self, board_states: np.ndarray
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Predict policy and value for a batch of board states
        Optimized for GPU batch processing
        Args:
            board_states: Board state array [batch_size, 14]
        Returns:
            policies: Move probabilities [batch_size, 6]
            values: Position evaluations [batch_size]
        """
        # Handle empty batch
        if len(board_states) == 0:
            return np.array([]), np.array([])

        # Convert to tensor and move to GPU
        x = torch.FloatTensor(board_states).to(self.device)

        log_policies, values = self(x)

        # Convert to numpy
        policies = torch.exp(log_policies).cpu().numpy()
        values = values.cpu().numpy().squeeze()

        # Handle single value case
        if values.ndim == 0:
            values = np.array([values])

        return policies, values

    def compile_for_inference(self):
        """
        Optional: Compile the model with TorchScript for faster inference
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.config.model.input_dim).to(self.device)

            # Trace the model
            traced_model = torch.jit.trace(self, dummy_input)

            print("Model successfully compiled with TorchScript")
            return traced_model
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing with standard PyTorch model")
            return self
