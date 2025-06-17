"""
Monte Carlo Tree Search implementation for Kalah
Optimized for the game's variable-length turns and tactical nature
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple
from collections import defaultdict
import threading


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.virtual_loss = 0

    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """Calculate UCB score for node selection"""
        if self.visit_count == 0:
            # Avoid division by zero, prioritize unvisited nodes
            return c_puct * self.prior * math.sqrt(parent_visits)

        exploitation = self.value()
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

        return exploitation + exploration

    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0


class MCTS:
    """
    Monte Carlo Tree Search for Kalah
    Handles variable-length turns and implements virtual loss for parallelization
    """

    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.nodes = {}
        self.lock = threading.Lock()

    def search(self, game, root_state: Optional[str] = None) -> np.ndarray:
        """
        Run MCTS simulations and return visit counts as action probabilities
        Args:
            game: KalahGame instance
            root_state: Optional root state key for tree reuse
        Returns:
            Action probabilities based on visit counts
        """
        if root_state is None:
            root_state = self._state_key(game)

        # Add Dirichlet noise to root node for exploration
        if root_state not in self.nodes:
            self._expand_node(game, root_state)

        root_node = self.nodes[root_state]
        valid_moves = game.get_valid_moves()

        # Add Dirichlet noise to root prior
        noise = np.random.dirichlet(
            [self.config.mcts.dirichlet_alpha] * len(valid_moves)
        )
        for action in range(len(valid_moves)):
            if valid_moves[action] and action in root_node.children:
                child = root_node.children[action]
                child.prior = (
                    1 - self.config.mcts.dirichlet_epsilon
                ) * child.prior + self.config.mcts.dirichlet_epsilon * noise[action]

        # Run simulations
        for _ in range(self.config.mcts.num_simulations):
            self._simulate(game.clone(), root_state)

        # Extract visit counts
        visits = np.zeros(6)
        for action, child in root_node.children.items():
            visits[action] = child.visit_count

        return visits

    def _simulate(self, game, root_state: str) -> float:
        """Run a single MCTS simulation"""
        path = []
        current_state = root_state

        # Selection phase - traverse tree until leaf
        while not game.game_over:
            if current_state not in self.nodes:
                # Expansion phase - expand leaf node
                self._expand_node(game, current_state)
                break

            node = self.nodes[current_state]

            # Check if we need to expand
            valid_moves = game.get_valid_moves()
            if not node.is_expanded():
                self._expand_node(game, current_state)
                break

            # Select best action
            action = self._select_action(node, valid_moves)
            if action is None:
                break

            # Apply virtual loss
            with self.lock:
                node.children[action].virtual_loss += 1

            path.append((current_state, action))

            # Make move
            extra_turn = game.make_move(action)
            current_state = self._state_key(game)

        # Evaluation phase
        if game.game_over:
            # Use actual game outcome
            value = game.get_reward(game.current_player)
        else:
            # Use neural network evaluation
            state = game.get_canonical_state()
            _, value = self.network.predict(state)

        # Backup phase - propagate value up the tree
        self._backup(path, value, game.current_player)

        return value

    def _expand_node(self, game, state_key: str):
        """Expand a leaf node by adding all valid actions"""
        if state_key in self.nodes:
            return

        # Get neural network predictions
        state = game.get_canonical_state()
        policy, value = self.network.predict(state)

        # Mask invalid actions and renormalize
        valid_moves = game.get_valid_moves()
        policy = policy * valid_moves
        policy_sum = np.sum(policy)

        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # All valid moves equally probable
            policy = valid_moves / np.sum(valid_moves)

        # Create node and children
        node = MCTSNode()
        for action in range(6):
            if valid_moves[action]:
                node.children[action] = MCTSNode(prior=policy[action])

        with self.lock:
            self.nodes[state_key] = node

    def _select_action(self, node: MCTSNode, valid_moves: np.ndarray) -> Optional[int]:
        """Select action using PUCT formula"""
        best_score = -float("inf")
        best_action = None

        parent_visits = max(1, node.visit_count)

        for action in range(6):
            if not valid_moves[action] or action not in node.children:
                continue

            child = node.children[action]
            score = child.ucb_score(parent_visits, self.config.mcts.c_puct)

            # Account for virtual loss
            if child.virtual_loss > 0:
                score -= child.virtual_loss

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _backup(self, path: list, value: float, final_player: int):
        """Backup value through the path"""
        current_player = final_player

        for state_key, action in reversed(path):
            node = self.nodes[state_key]
            child = node.children[action]

            # Update with lock for thread safety
            with self.lock:
                child.visit_count += 1
                child.virtual_loss -= 1

                # Negate value for opponent
                if current_player != final_player:
                    child.value_sum -= value
                else:
                    child.value_sum += value

                # Update parent node visit count
                node.visit_count += 1

            # Switch player (may not alternate due to extra turns)
            current_player = 1 - current_player

    def _state_key(self, game) -> str:
        """Generate unique key for game state"""
        return f"{game.current_player}:{game.board.tobytes().hex()}"

    def clear_tree(self):
        """Clear the search tree"""
        with self.lock:
            self.nodes.clear()

    def get_action_probabilities(self, game, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities for current position
        Args:
            game: Current game state
            temperature: Temperature for controlling exploration
        Returns:
            Action probabilities
        """
        visits = self.search(game)

        if temperature == 0:
            # Greedy selection
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)

        return probs
