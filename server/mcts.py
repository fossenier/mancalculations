"""
Monte Carlo Tree Search implementation for Kalah
Optimized for the game's variable-length turns and tactical nature
"""

import numpy as np
import numpy.typing as npt
import math
from typing import Dict, Optional, Tuple
from collections import defaultdict
import threading
from kalah_game import KalahGame


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(self, prior: float = 0.0) -> None:
        """
        Initializes a new instance of the class.

        Args:
            prior (float, optional): The prior probability or value associated with this node. Defaults to 0.0.

        Attributes:
            visit_count (int): The number of times this node has been visited.
            value_sum (float): The cumulative value from all visits to this node.
            prior (float): The prior probability or value for this node.
            children (dict): A dictionary mapping actions to child nodes.
            virtual_loss (int): The virtual loss used for parallelization in MCTS.
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.current_player = None
        self.children = {}
        self.virtual_loss = 0

    def value(self) -> float:
        """
        Calculates and returns the average value of the node.

        Returns:
            float: The average value (value_sum divided by visit_count) if visit_count is greater than 0,
                   otherwise 0.0.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """
        Calculates the Upper Confidence Bound (UCB) score for a node in Monte Carlo Tree Search (MCTS).

        The UCB score balances exploitation (the node's current value estimate) and
        exploration (the potential for discovering better outcomes). Unvisited nodes
        are prioritized by returning a high exploration value.

        Args:
            parent_visits (int): The total number of visits to the parent node.
            c_puct (float): The exploration constant that controls the balance between exploration and exploitation.

        Returns:
            float: The computed UCB score for the node.
        """
        if self.visit_count == 0:
            # Avoid division by zero, prioritize unvisited nodes
            return c_puct * self.prior * math.sqrt(parent_visits)

        exploitation = self.value()
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

        return exploitation + exploration

    def is_expanded(self) -> bool:
        """
        Checks if the current node has been expanded.

        Returns:
            bool: True if the node has one or more child nodes, False otherwise.
        """
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

    def search(
        self, game: KalahGame, root_state: Optional[str] = None
    ) -> npt.NDArray[np.float64]:
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
        root_node.current_player = game.current_player
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
        """
        Simulates a single Monte Carlo Tree Search (MCTS) rollout from the given root state.

        This method performs the following MCTS phases:
        1. Selection: Traverses the tree from the root state, selecting child
        nodes according to the tree policy, until a leaf node is reached.
        2. Expansion: Expands the leaf node by adding its children to the tree if
        it has not been expanded yet.
        3. Evaluation: Evaluates the leaf node using either the actual game outcome
        (if the game is over) or a neural network prediction.
        4. Backup: Propagates the evaluation result back up the path taken during
        selection, updating statistics for each node.

        Args:
            game: The game environment, which must provide methods for checking game over, getting valid moves, applying moves, and evaluating the state.
            root_state (str): The string representation of the root state from which to start the simulation.

        Returns:
            float: The value of the simulation from the perspective of the current player at the root state.
        """
        path = []
        current_state = root_state

        # Selection phase - traverse tree until leaf
        while True:
            if current_state not in self.nodes:
                self._expand_node(game, current_state)
                break

            node = self.nodes[current_state]
            node.current_player = game.current_player

            # Check if game is over BEFORE trying to expand or select
            if game.game_over:
                break

            valid_moves = game.get_valid_moves()
            if not node.is_expanded():
                self._expand_node(game, current_state)
                break

            # Check if any valid moves exist
            if not np.any(valid_moves):
                break

            action = self._select_action(node, valid_moves)
            if action is None:
                break

            with self.lock:
                node.children[action].virtual_loss += 1

            path.append(
                (current_state, action, game.current_player)
            )  # Store player explicitly

            # Make move and check for extra turn
            extra_turn = game.make_move(action)
            current_state = self._state_key(game)

            # Handle extra turns by continuing with same player context
            if extra_turn and not game.game_over:
                continue  # Don't alternate player expectation

            # Check game over after move
            if game.game_over:
                break

        # Evaluation phase
        if game.game_over:
            # Use actual game outcome
            value = game.get_reward(game.current_player)
        else:
            # Use neural network evaluation
            state = game.get_canonical_state()
            _, value = self.network.predict(state)

        # Backup phase - propagate value up the tree
        self._backup(path, value)

        return value

    def _expand_node(self, game: KalahGame, state_key: str) -> None:
        """
        Expands a node in the MCTS tree for the given game state.

        If the node corresponding to `state_key` already exists in the tree, the
        method returns immediately. Otherwise, it uses the neural network to predict
        the policy (action probabilities) and value for the current canonical game
        state. The policy is masked to zero out invalid moves and renormalized. If
        all moves are invalid according to the policy, valid moves are assigned
        equal probability. A new MCTSNode is created, and for each valid action,
        a child node is initialized with the corresponding prior probability. Finally,
        the new node is added to the tree in a thread-safe manner.

        Args:
            game: The game environment providing state and move information.
            state_key (str): A unique key representing the current game state.
        """
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
            if state_key not in self.nodes:
                self.nodes[state_key] = node

    def _select_action(
        self, node: MCTSNode, valid_moves: npt.NDArray[np.bool_]
    ) -> Optional[int]:
        """
        Selects the best action from the given node based on the UCB (Upper Confidence Bound)
        score, considering only valid moves and accounting for virtual loss.

        Args:
            node (MCTSNode): The current node in the Monte Carlo Tree Search.
            valid_moves (np.ndarray): A boolean or integer array indicating which moves are valid (typically of length 6).

        Returns:
            Optional[int]: The index of the best action to take, or None if no valid action is found.
        """
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

    def _backup(self, path: list, value: float) -> None:
        """
        Propagates the simulation result back through the path of visited nodes, updating visit counts and value sums.

        Args:
            path (list): A list of (state_key, action) tuples representing the sequence of nodes and actions taken during the simulation.
            value (float): The simulation result to be backed up, typically from the perspective of the final player.

        Notes:
            - Increments visit counts for each node and child along the path.
            - Adjusts virtual loss for each child node.
            - Updates value sums based on whether the child node's current player matches the final player.
        """
        # Get the final player from the last node in the path
        final_state_key = path[-1][0] if path else None
        final_player = (
            self.nodes[final_state_key].current_player if final_state_key else None
        )

        for state_key, action, _ in reversed(path):
            node = self.nodes[state_key]
            child = node.children[action]

            with self.lock:
                child.visit_count += 1
                child.virtual_loss = max(0, child.virtual_loss - 1)

                # Value from perspective of the player who made this move
                # The child node's current_player is who benefits from this move's outcome
                node_value = value if child.current_player == final_player else -value
                child.value_sum += node_value
                node.visit_count += 1

    def _state_key(self, game: KalahGame) -> str:
        """
        Generate a unique string key representing the current state of the game.

        The key is composed of the current player's identifier and a hexadecimal
        representation of the canonical game state. This is useful for hashing or
        caching game states in algorithms such as Monte Carlo Tree Search (MCTS).

        Args:
            game (KalahGame): The current game instance.

        Returns:
            str: A unique string key for the given game state.
        """
        return f"{game.current_player}:{game.get_canonical_state().tobytes().hex()}"

    def clear_tree(self) -> None:
        """
        Clears all nodes from the MCTS tree in a thread-safe manner.

        This method acquires a lock to ensure that the operation is safe in multi-threaded environments,
        then removes all nodes from the internal node storage.
        """
        with self.lock:
            self.nodes.clear()

    def get_action_probabilities(
        self, game: KalahGame, temperature: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """
        Calculates the probability distribution over actions based on visit counts from MCTS search.
        Args:
            game (KalahGame): The current game state for which to compute action probabilities.
            temperature (float, optional): Controls the level of exploration. A value close to 0 selects the most visited action deterministically, while higher values increase exploration. Defaults to 1.0.
        Returns:
            npt.NDArray[np.float64]: A probability distribution over possible actions, summing to 1.0.
        """
        visits = self.search(game)

        # Add small constant to avoid zero division
        visits = visits + 1e-8

        if temperature == 0 or temperature < 1e-8:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = np.power(visits, 1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)

        return probs
