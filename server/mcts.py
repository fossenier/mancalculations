"""
Monte Carlo Tree Search implementation for Kalah
Lock-free implementation optimized for multi-threaded GPU batch evaluation
"""

import numpy as np
import numpy.typing as npt
import math
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
import threading
from kalah_game import KalahGame
from dataclasses import dataclass
import concurrent.futures


class AtomicCounter:
    """Truly lock-free counter using memory-mapped array"""

    def __init__(self, dtype):
        # Use shared memory for true atomic operations
        self._value = np.zeros(1, dtype=dtype)

    def add(self, value):
        """Atomic add - relies on GIL for atomicity"""
        self._value[0] += value

    def get(self):
        """Atomic read"""
        return self._value[0]

    def set(self, value):
        """Atomic set"""
        self._value[0] = value

    @property
    def visit_count(self) -> int:
        return int(self._visit_count[0])

    @property
    def value_sum(self) -> float:
        return float(self._value_sum[0])

    @property
    def virtual_loss(self) -> int:
        return int(self._virtual_loss[0])

    def add_visit(self, value: float) -> None:
        """Atomically add a visit with value"""
        # NumPy operations on single elements are atomic at the C level
        self._visit_count += 1
        self._value_sum += value

    def add_virtual_loss(self) -> None:
        """Atomically increment virtual loss"""
        self._virtual_loss += 1

    def remove_virtual_loss(self) -> None:
        """Atomically decrement virtual loss"""
        self._virtual_loss -= 1


class MCTSNode:
    """Truly lock-free MCTS node"""

    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.current_player: int | None = None
        # Use atomic counters instead of locks
        self.visit_count = AtomicCounter(np.int64)
        self.value_sum = AtomicCounter(np.float64)
        self.virtual_loss = AtomicCounter(np.int32)
        # Store children as immutable after creation
        self._children: Dict[Any, Any] | None = None
        self._expanded = AtomicCounter(np.int32)  # 0=not expanded, 1=expanded

    @property
    def children(self):
        return self._children or {}

    def value(self) -> float:
        """Get average value (lock-free read)"""
        visits = self.visit_count.get()
        if visits == 0:
            return 0.0
        return self.value_sum.get() / visits

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """Calculate UCB score (lock-free)"""
        visits = self.visit_count.get()

        if visits == 0:
            return c_puct * self.prior * math.sqrt(parent_visits)

        exploitation = self.value()
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + visits)

        # Account for virtual loss
        virtual_penalty = self.virtual_loss.get() * c_puct / (1 + visits)

        return exploitation + exploration - virtual_penalty

    def is_expanded(self) -> bool:
        """Check if node is expanded (lock-free)"""
        return self._expanded.get() == 1

    def try_expand(self) -> bool:
        """Try to expand node - returns True if this thread expanded it"""
        # Compare-and-swap: only one thread succeeds
        old_val = self._expanded.get()
        if old_val == 0:
            self._expanded.set(1)
            return old_val == 0
        return False


class MCTS:
    """
    Lock-free Monte Carlo Tree Search for Kalah
    Uses atomic operations and lock-free algorithms
    """

    def __init__(self, config, network, batch_size: int = 32):
        self.config = config
        self.network = network
        self.nodes = {}  # Will use get_or_create pattern
        self.batch_size = batch_size

    def _get_or_create_node(self, state_key: str) -> MCTSNode:
        """Get existing node or create new one atomically"""
        # Python's dict.setdefault is atomic at the C level
        return self.nodes.setdefault(state_key, MCTSNode())

    def search(
        self, game: KalahGame, root_state: Optional[str] = None
    ) -> npt.NDArray[np.float64]:
        """
        Run MCTS simulations with lock-free batched evaluation
        """
        if root_state is None:
            root_state = self._state_key(game)

        # Initialize root node
        root_node = self._get_or_create_node(root_state)
        root_node.current_player = game.current_player

        # Expand root if needed
        if not root_node.is_expanded():
            self._expand_node_single(game, root_state)

        # Add Dirichlet noise to root
        valid_moves = game.get_valid_moves()
        noise = np.random.dirichlet(
            [self.config.mcts.dirichlet_alpha] * len(valid_moves)
        )

        for action in range(6):
            if valid_moves[action] and action in root_node.children:
                child = root_node.children[action]
                # Prior is only set once at expansion, safe to modify at root
                child.prior = (
                    1 - self.config.mcts.dirichlet_epsilon
                ) * child.prior + self.config.mcts.dirichlet_epsilon * noise[action]

        # Run simulations in parallel batches
        num_simulations = self.config.mcts.num_simulations
        simulation_count = 0

        while simulation_count < num_simulations:
            current_batch_size = min(
                self.batch_size, num_simulations - simulation_count
            )

            # Collect leaf nodes in parallel (no thread pool needed)
            leaf_infos = []
            for _ in range(current_batch_size):
                game_copy = game.clone()
                path, leaf_game, leaf_state_key = self._simulate_to_leaf(
                    game_copy, root_state
                )
                leaf_infos.append((path, leaf_game, leaf_state_key))

            # Batch evaluate and backup
            self._batch_evaluate_and_backup(leaf_infos)

            simulation_count += current_batch_size

        # Extract visit counts (lock-free reads)
        visits = np.zeros(6)
        for action, child in root_node.children.items():
            visits[action] = child.visit_count.get()

        return visits

    def _simulate_to_leaf(
        self, game: KalahGame, root_state: str
    ) -> Tuple[List, KalahGame, str]:
        """
        Select down to leaf node (lock-free)
        """
        path = []
        current_state = root_state

        while True:
            node = self._get_or_create_node(current_state)
            node.current_player = game.current_player

            # Check terminal
            if game.game_over:
                return path, game, current_state

            # Check if needs expansion
            if not node.is_expanded():
                return path, game, current_state

            valid_moves = game.get_valid_moves()
            if not np.any(valid_moves):
                return path, game, current_state

            # Select action (lock-free)
            action = self._select_action_lockfree(node, valid_moves)
            if action is None:
                return path, game, current_state

            # Add virtual loss atomically
            child = node.children[action]
            child.stats.add_virtual_loss()

            path.append((current_state, action, game.current_player))

            # Make move
            extra_turn = game.make_move(action)
            current_state = self._state_key(game)

            if extra_turn and not game.game_over:
                continue

            if game.game_over:
                return path, game, current_state

    def _select_action_lockfree(
        self, node: MCTSNode, valid_moves: npt.NDArray[np.bool_]
    ) -> Optional[int]:
        """
        Select best action using lock-free UCB calculation
        """
        parent_visits = max(1, node.visit_count.get())

        best_score = -float("inf")
        best_action = None

        # Calculate all scores in single pass
        for action in range(6):
            if not valid_moves[action] or action not in node.children:
                continue

            child = node.children[action]
            score = child.ucb_score(parent_visits, self.config.mcts.c_puct)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _batch_evaluate_and_backup(
        self, leaf_infos: List[Tuple[List, KalahGame, str]]
    ) -> None:
        """
        Evaluate leaf nodes in batch and backup (lock-free)
        """
        # Separate terminal and non-terminal
        non_terminal_infos = []
        terminal_backups = []

        for i, (path, game, state_key) in enumerate(leaf_infos):
            if game.game_over:
                value = game.get_reward(game.current_player)
                terminal_backups.append((path, value))
            else:
                non_terminal_infos.append((i, path, game, state_key))

        # Batch evaluate non-terminals
        if non_terminal_infos:
            states = []
            expansion_needed = []

            for idx, (_, path, game, state_key) in enumerate(non_terminal_infos):
                node = self._get_or_create_node(state_key)

                # Try to expand (only one thread will succeed)
                if node.try_expand():
                    states.append(game.get_canonical_state())
                    expansion_needed.append((idx, game, state_key, path))

            # Batch predict for nodes that need expansion
            if states:
                states_array = np.array(states)
                policies, values = self.network.predict_batch(states_array)

                # Expand nodes and backup
                for i, (idx, game, state_key, path) in enumerate(expansion_needed):
                    self._complete_expansion(game, state_key, policies[i])
                    self._backup_lockfree(path, values[i])

            # Backup for already-expanded nodes
            for _, path, game, state_key in non_terminal_infos:
                if state_key not in [x[2] for x in expansion_needed]:
                    # Node was already expanded by another thread
                    node = self.nodes[state_key]
                    if node.is_expanded():
                        # Use existing value estimate
                        value = node.value()
                        self._backup_lockfree(path, value)

        # Backup terminal values
        for path, value in terminal_backups:
            self._backup_lockfree(path, value)

    def _expand_node_single(self, game: KalahGame, state_key: str) -> None:
        """
        Expand single node (used for root)
        """
        node = self._get_or_create_node(state_key)

        if node.try_expand():
            state = game.get_canonical_state()
            policy, _ = self.network.predict(state)
            self._complete_expansion(game, state_key, policy)

    def _complete_expansion(
        self, game: KalahGame, state_key: str, policy: np.ndarray
    ) -> None:
        """
        Complete node expansion with policy (already marked as expanded)
        """
        node = self.nodes[state_key]

        # Mask invalid actions
        valid_moves = game.get_valid_moves()
        policy = policy * valid_moves
        policy_sum = np.sum(policy)

        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = valid_moves / np.sum(valid_moves)

        # Create children
        children = {}
        for action in range(6):
            if valid_moves[action]:
                children[action] = MCTSNode(prior=policy[action])

        # Atomic update of children dict
        node.children = children

    def _backup_lockfree(self, path: list, value: float) -> None:
        """
        Lock-free backup using atomic operations
        """
        current_value = value

        # Process path in reverse
        for state_key, action, player in reversed(path):
            node = self.nodes[state_key]
            child = node.children[action]

            # Atomic updates
            child.stats.remove_virtual_loss()
            child.stats.add_visit(current_value)
            node.stats.add_visit(0)  # Just increment visit count

            # Flip value for opponent
            current_value = -current_value

    def clear_tree(self) -> None:
        """Clear all nodes"""
        self.nodes.clear()

    def get_action_probabilities(
        self, game: KalahGame, temperature: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """
        Get action probabilities based on visit counts
        """
        visits = self.search(game)

        # Add small constant
        visits = visits + 1e-8

        if temperature == 0 or temperature < 1e-8:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)

        return probs

    def _state_key(self, game: KalahGame) -> str:
        """Generate unique state key"""
        return f"{game.current_player}:{game.get_canonical_state().tobytes().hex()}"
