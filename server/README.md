# AlphaZero Kalah - Superhuman AI Training on 4x A100 GPUs

A complete implementation of AlphaZero for Kalah (6-stone Mancala) optimized for multi-GPU training on NVIDIA A100 hardware.

## Overview

This implementation follows the AlphaZero algorithm to train a superhuman Kalah AI through self-play reinforcement learning. The system is designed to efficiently utilize 4x NVIDIA A100 GPUs for distributed training while generating self-play games on CPU workers.

## Features

- **Distributed Training**: Optimized for 4x A100 GPUs with data parallel training
- **Mixed Precision**: FP16/TF32 training for maximum GPU utilization
- **Efficient MCTS**: Monte Carlo Tree Search with virtual loss for parallelization
- **Comprehensive Evaluation**: Tournament play against various opponents
- **Real-time Monitoring**: Web interface and logging for tracking progress
- **Checkpoint Management**: Automatic model saving and recovery

## System Requirements

- 4x NVIDIA A100 GPUs (40GB or 80GB)
- 64+ CPU cores for self-play generation
- 256GB+ system RAM
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.8+
- CUDA 11.0+
- NCCL for multi-GPU communication

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/alphazero-kalah.git
cd alphazero-kalah
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Start training with default configuration:

```bash
python main.py
```

Resume from checkpoint:

```bash
python main.py --resume ./kalah_checkpoints/kalah_alphazero_iter100.pt
```

### Monitoring

Start the web monitoring interface:

```bash
python monitor.py --web --port 5000
```

Then open http://localhost:5000 in your browser.

Generate training plots:

```bash
python monitor.py --plot
```

Generate a training report:

```bash
python monitor.py --report training_report.md
```

### Evaluation

Evaluate a specific model:

```bash
python main.py --evaluate ./kalah_checkpoints/kalah_alphazero_best.pt
```

## Configuration

The system uses a comprehensive configuration in `config.py`. Key parameters:

### Model Architecture

- **Residual Blocks**: 20 (15-25 recommended)
- **Filters**: 256 (256-512 recommended)
- **Kernel Sizes**: [3, 5, 7] for multi-scale analysis

### MCTS Parameters

- **Simulations**: 1200 per move (800-1600 recommended)
- **c_puct**: 2.0 (1.5-2.5 for Kalah's tactical nature)
- **Temperature**: 1.0 → 0.1 after 30 moves

### Training Parameters

- **Batch Size**: 2048 per GPU (8192 total)
- **Learning Rate**: 0.001 with cosine annealing
- **Replay Buffer**: 1M positions
- **Games per Iteration**: 30,000

### Custom Configuration

Create a custom config file:

```json
{
  "model": {
    "num_residual_blocks": 25,
    "num_filters": 512
  },
  "mcts": {
    "num_simulations": 1600,
    "c_puct": 2.2
  },
  "training": {
    "learning_rate": 0.0005
  }
}
```

Use it for training:

```bash
python main.py --config custom_config.json
```

## File Structure

```
alphazero-kalah/
├── config.py           # Configuration management
├── kalah_game.py       # Kalah game implementation
├── network.py          # Neural network architecture
├── mcts.py            # Monte Carlo Tree Search
├── trainer.py         # Distributed training system
├── self_play.py       # Self-play game generation
├── evaluator.py       # Model evaluation
├── main.py            # Main training loop
├── monitor.py         # Monitoring and visualization
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Training Progress

Expected performance milestones:

| Time      | Milestone            | Win Rate vs Minimax-5 |
| --------- | -------------------- | --------------------- |
| 12-24h    | Basic competent play | 60-80%                |
| 2-4 days  | Strong amateur level | 85-95%                |
| 1-2 weeks | Near-optimal play    | 95-99%                |

## Interacting with the Training System

### Real-time Monitoring

The system provides multiple ways to monitor training:

1. **Console Output**: Detailed logs in `./kalah_logs/`
2. **TensorBoard**:
   ```bash
   tensorboard --logdir ./kalah_tensorboard
   ```
3. **Web Dashboard**: Real-time stats at http://localhost:5000
4. **JSON API**: GET http://localhost:5000/api/stats

### Checkpoint Management

Models are saved every 10 iterations by default:

- Regular checkpoints: `./kalah_checkpoints/kalah_alphazero_iter{N}.pt`
- Best model: `./kalah_checkpoints/kalah_alphazero_best.pt`
- Training state: `./kalah_checkpoints/training_state.json`

### Accessing Saved Models

Load a checkpoint in Python:

```python
import torch
from network import KalahNetwork
from config import get_config

# Load configuration
config = get_config()

# Create network
network = KalahNetwork(config)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pt')
network.load_state_dict(checkpoint['model_state_dict'])

# Use the model
state = game.get_canonical_state()
policy, value = network.predict(state)
```

## Advanced Usage

### Multi-Node Training

For training across multiple nodes:

```bash
# On master node
export MASTER_ADDR=master_ip
export MASTER_PORT=12355
python main.py --num-gpus 8

# On worker nodes
export MASTER_ADDR=master_ip
export MASTER_PORT=12355
python main.py --num-gpus 8
```

### Custom Opponents

Add custom opponents in `evaluator.py`:

```python
class CustomPlayer:
    def __init__(self):
        self.name = "Custom"

    def get_action(self, game):
        # Your logic here
        return action
```

### Experiment Tracking

Enable Weights & Biases integration:

```python
# In trainer.py
import wandb
wandb.init(project="alphazero-kalah")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Self-Play**

   - Increase CPU workers
   - Reduce MCTS simulations
   - Enable tree reuse

3. **Training Instability**
   - Lower learning rate
   - Increase gradient clipping
   - Check replay buffer diversity

### Performance Optimization

1. **GPU Utilization**

   - Monitor with `nvidia-smi`
   - Target 90%+ utilization
   - Adjust batch size accordingly

2. **CPU-GPU Balance**
   - Ensure self-play generates data faster than training consumes
   - Use async data loading
   - Profile with `cProfile`

## Citation

If you use this implementation, please cite:

```bibtex
@software{alphazero_kalah,
  title={AlphaZero Kalah: Superhuman AI Training on 4x A100 GPUs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/alphazero-kalah}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Based on the AlphaZero paper by Silver et al. (2017)
- Kalah game implementation inspired by various open-source projects
- Optimizations for A100 GPUs based on NVIDIA best practices
