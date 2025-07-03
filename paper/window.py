import numpy as np
import numpy.typing as npt
from config import AlphaZeroConfig
from mcts import MCTSStatistic
from multiprocessing import Queue
from typing import List


CONFIG = AlphaZeroConfig()


class Window:
    def __init__(self, games: Queue, size=CONFIG.window_size):
        # Queue that workers put finished games in
        self.games = games

        # Revolving window of latest X games for sampling
        self.states = np.zeros((size, 14), dtype=np.float32)
        self.policies = np.zeros((size, 6), dtype=np.float32)
        self.values = np.zeros((size, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

        # TODO: to boot up training fill with copied randomness

    def run(self):
        """
        Main loop - call this to start consuming from the queue.
        This will run forever until interrupted.
        """
        while True:
            try:
                # Use a short timeout to avoid busy waiting
                games: List[List[MCTSStatistic]] = self.games.get(timeout=0.1)
                # Each put() will set an array of MCTSStatistics
                for game in games:
                    print("New complete game: ")
                    print(game[-1].state)

                    for move in game:
                        state = move.state
                        policy = move.move_probabilities
                        value = move.value
                        self._add(state, policy, value)
            except Exception:
                # Timeout or empty queue, just continue
                continue

    def _add(self, state, policy, value):
        """
        Pop another game in the window
        """
        pos = self.idx % len(self.states)
        self.states[pos] = state
        self.policies[pos] = policy
        self.values[pos] = value
        self.idx += 1
        self.size = min(self.size + 1, len(self.states))

    def sample(self, batch_size):
        """
        Sample uniformly across all positions in the buffer.
        """
        if self.size == 0:
            raise ValueError("No data to sample from.")
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch_states = self.states[indices]
        batch_policies = self.policies[indices]
        batch_values = self.values[indices]
        return batch_states, batch_policies, batch_values
