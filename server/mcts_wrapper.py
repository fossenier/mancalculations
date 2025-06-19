"""
Python wrapper for C++ MCTS implementation
Provides drop-in replacement for the Python MCTS class
"""

import numpy as np
from typing import Optional, Dict, Any
import os
import sys

# Import the C++ module
try:
    import mcts_cpp
except ImportError:
    print("C++ MCTS module not found. Please build it first using:")
    print("  mkdir build && cd build")
    print("  cmake ../mcts")
    print("  make -j")
    print("  cp mcts_cpp*.so ../")
    sys.exit(1)


class KalahGameWrapper:
    """Wrapper to make C++ KalahGame compatible with Python code"""

    def __init__(self, cpp_game=None):
        self._game = cpp_game or mcts_cpp.KalahGame()

    def make_move(self, action: int) -> bool:
        return self._game.make_move(action)

    def get_valid_moves(self) -> np.ndarray:
        return np.array(self._game.get_valid_moves())

    def get_canonical_state(self) -> np.ndarray:
        return np.array(self._game.get_canonical_state())

    def get_reward(self, player: int) -> float:
        return self._game.get_reward(player)

    def clone(self):
        return KalahGameWrapper(self._game.clone())

    @property
    def game_over(self) -> bool:
        return self._game.game_over

    @property
    def current_player(self) -> int:
        return self._game.current_player


class MCTSWrapper:
    """Python wrapper for C++ MCTS that matches the original Python interface"""

    def __init__(self, config: Dict[str, Any], network=None, batch_size: int = 32):
        """
        Initialize C++ MCTS with Python config

        Args:
            config: AlphaZeroConfig object or dict
            network: Neural network (will extract model path)
            batch_size: Batch size for GPU inference (ignored, C++ uses its own)
        """
        # Convert config to dict if needed
        if hasattr(config, "__dict__"):
            config_dict = self._config_to_dict(config)
        else:
            config_dict = config

        # Get model path from network if available
        model_path = ""
        if hasattr(network, "model_path"):
            model_path = network.model_path
        elif hasattr(network, "save_path"):
            model_path = network.save_path

        # Detect number of GPUs
        import torch

        num_gpus = torch.cuda.device_count()

        # Get number of CPU cores
        num_threads = min(64, os.cpu_count() or 64)

        # Create C++ MCTS instance
        self._mcts = mcts_cpp.MCTS(config_dict, model_path, num_threads, num_gpus)

        print(f"Initialized C++ MCTS with {num_threads} threads and {num_gpus} GPUs")

    def _config_to_dict(self, config) -> Dict[str, Any]:
        """Convert config object to nested dict"""
        result = {}

        # Extract MCTS config
        if hasattr(config, "mcts"):
            mcts_config = {}
            for attr in [
                "num_simulations",
                "c_puct",
                "dirichlet_alpha",
                "dirichlet_epsilon",
                "temperature_threshold",
                "initial_temperature",
                "final_temperature",
            ]:
                if hasattr(config.mcts, attr):
                    mcts_config[attr] = getattr(config.mcts, attr)
            result["mcts"] = mcts_config

        return result

    def search(self, game, root_state: Optional[str] = None) -> np.ndarray:
        """
        Run MCTS simulations

        Args:
            game: KalahGame instance (Python or C++)
            root_state: Optional root state key

        Returns:
            Visit counts for each action
        """
        # Convert Python game to C++ if needed
        if not isinstance(game, mcts_cpp.KalahGame):
            cpp_game = self._convert_game(game)
        else:
            cpp_game = game

        # Run search
        visits = self._mcts.search(cpp_game, root_state or "")
        return np.array(visits)

    def get_action_probabilities(self, game, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities based on visit counts

        Args:
            game: KalahGame instance
            temperature: Temperature for action selection

        Returns:
            Action probability distribution
        """
        # Convert Python game to C++ if needed
        if not isinstance(game, mcts_cpp.KalahGame):
            cpp_game = self._convert_game(game)
        else:
            cpp_game = game

        probs = self._mcts.get_action_probabilities(cpp_game, temperature)
        return np.array(probs)

    def clear_tree(self) -> None:
        """Clear the MCTS tree"""
        self._mcts.clear_tree()

    def _convert_game(self, python_game):
        """Convert Python KalahGame to C++ KalahGame"""
        cpp_game = mcts_cpp.KalahGame()

        # Copy game state
        # This requires implementing proper state transfer
        # For now, replay all moves
        if hasattr(python_game, "move_history"):
            for move in python_game.move_history:
                cpp_game.make_move(move)

        return cpp_game

    def _state_key(self, game) -> str:
        """Generate state key (for compatibility)"""
        if hasattr(game, "get_canonical_state"):
            state = game.get_canonical_state()
        else:
            state = np.array(game.get_canonical_state())

        return f"{game.current_player}:{state.tobytes().hex()}"


# Make it a drop-in replacement
MCTS = MCTSWrapper


# Build script
def build_cpp_module():
    """Build the C++ module"""
    import subprocess
    import os

    print("Building C++ MCTS module...")

    # Create build directory
    os.makedirs("build", exist_ok=True)

    # Run cmake
    subprocess.run(["cmake", ".."], cwd="build", check=True)

    # Build with all cores
    num_cores = os.cpu_count() or 1
    subprocess.run(["make", f"-j{num_cores}"], cwd="build", check=True)

    # Copy the module
    import glob

    so_files = glob.glob("build/mcts_cpp*.so")
    if so_files:
        import shutil

        shutil.copy(so_files[0], ".")
        print(f"Successfully built and copied {so_files[0]}")
    else:
        print("Error: Could not find built module")


if __name__ == "__main__":
    build_cpp_module()
