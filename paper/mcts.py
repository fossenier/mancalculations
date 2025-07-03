"""
Monte Carlo Tree Search (MCTS) implemented for Mancala (Kalah) based on the
AlphaZero paper. Written to be multi-threaded across soarserver's 64 threads,
and to pass the work to the workers handling 4x A100 interaction. Not currently
setup to support virtual loss.
"""

import math
import numpy as np
from config import AlphaZeroConfig
from kalah import KalahGame
from numpy.typing import NDArray
from typing import List, Tuple, cast


class MCTSNode:
    """
    Node for a MCTS tree.
    """

    def __init__(
        self,
        player: int | None,
        action: int | None,
        prior: np.float32 | None,
        parent: "MCTSNode | None",
    ) -> None:
        """
        Initializes a new MCTSNode with the given player, action, prior probability,
        and parent node.

        An expanded child node will not have a player, and a root node will not have
        an action or prior probability or parent.
        """
        self.action: int | None = action  # Pit moved from to reach this state
        self.visit_count: int = 0  # N = visit count
        self.total_value: np.float32 = np.float32(0.0)  # W = total value
        # Q = W/N = average value = state value of the node
        self.probability: np.float32 | None = (
            prior  # P = probability of taking this action
        )
        # (as determined by the policy network when given the parent state)
        self.player: int | None = player  # Player who made the move to reach this state
        self.children: dict[int, "MCTSNode"] = {}  # Child nodes, keyed by action
        self.parent: "MCTSNode | None" = parent  # Parent node, None for root node

    def apply_dirichlet_noise(
        self, root_dirichlet_alpha: float, root_exploration_fraction: float
    ) -> None:
        actions = self.children.keys()
        noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
        frac = root_exploration_fraction
        for a, n in zip(actions, noise):
            if self.children[a].probability is None:
                raise ValueError(
                    "Cannot apply Dirichlet noise to a child node without a prior probability."
                )

            self.children[a].probability = np.float32(
                np.float32(self.children[a].probability) * (1 - frac) + n * frac
            )

    def expand(
        self, prior: NDArray[np.float32], valid_actions_mask: NDArray[np.float32]
    ) -> None:
        """
        Expands the node by adding children for each valid action with a prior probability.
        This is called when the node is a leaf node and needs to be expanded.
        """
        if self.expanded():
            raise ValueError("Cannot expand an already expanded node.")

        # valid_actions is a mask (e.g., [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) over actions [0-5]
        for action, is_valid in enumerate(valid_actions_mask):
            if is_valid:
                self.children[action] = MCTSNode(
                    player=None, action=action, prior=prior[action], parent=self
                )

    def expanded(self) -> bool:
        """
        Returns True if the node has been expanded (i.e., has children).
        """
        return len(self.children) > 0

    def value(self) -> np.float32:
        """
        Returns the value of the node, which is the average value.
        """
        if self.visit_count == 0:
            return np.float32(0.0)
        return self.total_value / self.visit_count


class MCTSStatistic:
    """
    The data needed to track each root state in a MCTS search for
    use by the model to get trained.
    """

    def __init__(
        self, game: KalahGame, move_probabilities: NDArray[np.float32]
    ) -> None:
        # Data for model
        self.state = game.get_canonical_state()  # Stays as is
        self.move_probabilities: NDArray[np.float32] = move_probabilities
        self.value: float = 0.0

        # Needed to assign value once game reaches terminal state
        self.player_to_move: int = game.current_player

    def set_value(self, game_winner: int) -> None:
        """
        Sets the value of the game based on the winner.
        If the player to move is the winner, set value to 1.0,
        if the player to move is the loser, set it to -1.0,
        and if the game is a draw, set it to 0.0.
        """
        if self.player_to_move == game_winner:
            self.value = 1.0
        elif game_winner == -1:  # Draw
            self.value = 0.0
        else:
            self.value = -1.0


class MCTS:
    def __init__(self) -> None:
        """
        Sets up all variables needed to run the MCTS search, and also puts in
        a model request for the base root state to get the queue started.
        """
        # To make model calls, and save the game when complete

        # True game state for MCTS flow
        self.game = KalahGame()
        # Config to control MCTS behaviour
        self.config = AlphaZeroConfig()
        # How many simulations + model calls before choosing an action (and new root)
        self.max_simulations: int = self.config.num_simulations
        # The current MCTS root node, initially None
        self.root: MCTSNode | None = None
        # The real actions taken during the course of the game
        self.chosen_path: list[MCTSStatistic] = []
        # The latest simulation to leaf node, to backpropogate
        self.latest_sim_path: list[MCTSNode] = []
        self.latest_valid_actions_mask: NDArray[np.float32] = np.zeros(
            6, dtype=np.float32
        )

        self.first_call = self.game.get_canonical_state()

    def step(
        self, prior: NDArray[np.float32], value: np.float32
    ) -> Tuple[NDArray[np.float32] | None, List[MCTSStatistic] | None]:
        """
        If returning NDArray[np.float32], that is a model request and the simulation is still running.
        If returning List[MCTSStatistic], that is a window post and the game has ended, make a new one.
        Returns True if the game is still running, the NDArray[np.float32] will return (model request)
        False if it has ended, the List[MCTSStatistic] will return (window post).

        When a game ends, it is pushed to the window queue.
        """
        # Start by backpropogating (creates root node when needed!)
        self.backpropagate(prior, value, self.game.current_player)
        self.latest_sim_path.clear()

        # If the root node has been visited enough times, choose a real action
        if self.root is None:
            raise ValueError("Root node is None, cannot step MCTS.")
        if self.root.visit_count >= self.max_simulations:
            # This goes to a new root node, save the current one first
            self.chosen_path.append(
                MCTSStatistic(
                    game=self.game,
                    move_probabilities=np.array(
                        [
                            child.visit_count / self.root.visit_count
                            for child in self.root.children.values()
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            self.perform_action()
            # When the game ends, prepare statistics and push them to the window
            if self.game.game_over:
                self.prepare_statistics()
                # print(self.chosen_path)
                # self.window.put(self.chosen_path)
                # self.window.empty()  # Flush the window queue
                return (None, self.chosen_path)
            # if self.game.game_over:
            #     self.prepare_statistics()
            #     print(
            #         f"Before window.put: chosen_path length = {len(self.chosen_path)}"
            #     )
            #     print(f"chosen_path contents: {self.chosen_path}")

            #     # Create a copy to avoid reference issues
            #     path_copy = list(self.chosen_path)
            #     print(f"path_copy length = {len(path_copy)}")

            #     self.window.put(path_copy)
            #     print("Successfully put data in window")
            #     print("true" if not self.window.empty() else "false")
            #     return False  # Game is over, return False
            else:
                return (
                    self.game.get_canonical_state(),
                    None,
                )  # Game continues (new root node will be created next time)

        # Otherwise, run another simulation for this root node
        return (self.simulate(), None)

    def backpropagate(
        self, prior: NDArray[np.float32], value: np.float32, player: int
    ) -> None:
        """
        Backpropogates the model's value prediction up the latest visited path
        (and updates visit counts + values too).
        """
        # No nodes in the latest chosen path implies this will be a root node
        root_created = False
        if not self.latest_sim_path:
            self.root = MCTSNode(action=None, player=player, prior=None, parent=None)
            # Set the root in the sim path to get expanded
            root_created = True
            self.latest_sim_path.append(self.root)
            self.latest_valid_actions_mask = self.game.get_valid_moves()

        # Backpropagate the value up the tree
        # First, expand the leaf node
        # NOTE: This shouldn't attempt to expand the terminal state, due to the
        # control flow in step(), but it will be safe and produce no children
        # if that happens somehow (due to expand()'s use of get_valid_moves())
        leaf_node = self.latest_sim_path[-1]
        # if root_created:
        # print(self.game.get_state())
        # print(f"Expanding root node with prior probabilities {prior}")
        leaf_node.expand(prior, self.latest_valid_actions_mask)

        # Apply Dirichlet noise to the root node's children if the root was just made
        if root_created:
            leaf_node.apply_dirichlet_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        for node in reversed(self.latest_sim_path):
            node.visit_count += 1
            # Value is negated for the opponent
            node.total_value += value if node.player == player else -value

    def perform_action(self) -> None:
        """
        Chooses a true action from the current root node, and updates the
        game state.
        """
        # Shouldn't happen, but just in case
        if self.root is None:
            raise ValueError("Root node is None, cannot perform action.")

        # NOTE: we don't need to mask for valid actions as only valid children are created
        # The number of moves to play a little more stochastically
        if len(self.chosen_path) < self.config.num_sampling_moves:
            probabilities = []
            for action, child in self.root.children.items():
                probabilities.append(
                    (
                        action,
                        (child.visit_count ** (1.0 / self.config.temperature))
                        / (
                            self.root.visit_count ** (1.0 / self.config.temperature) - 1
                        ),
                    )
                )
            # Normalize probabilities
            total_prob = sum(prob for _, prob in probabilities)
            if total_prob > 0:
                probabilities = [
                    (action, prob / total_prob) for action, prob in probabilities
                ]
            else:
                # fallback: uniform distribution if all probs are zero
                num_actions = len(probabilities)
                probabilities = [
                    (action, 1.0 / num_actions) for action, _ in probabilities
                ]
            # print(
            #     f"Sampling action from root with {len(probabilities)} children, total prob: {total_prob}"
            # )
            action = np.random.choice(
                [action for action, _ in probabilities],
                p=[prob for _, prob in probabilities],
            )
        # Be deterministic, choose the child that was most visited
        else:
            # Select the action (key) with the highest visit count
            # print(
            #     f"Choosing action from root with {len(self.root.children)} children, visit counts: {[child.visit_count for child in self.root.children.values()]}"
            # )
            action = max(
                self.root.children.items(), key=lambda item: item[1].visit_count
            )[0]

        # Update the game state

        self.game.make_move(int(action))

    def prepare_statistics(self) -> None:
        # TODO: prepare self.chosen_path for the window
        pass

    def simulate(self) -> NDArray[np.float32] | None:
        # Grab the moment in time of the root, copy it, save it to the sim path
        node = cast(MCTSNode, self.root)
        scratch_game = self.game.clone()
        self.latest_sim_path.append(node)

        # Run down the tree to a leaf node, selecting the child based on PUCT
        while node.expanded():
            action, node = self.select_child(node)
            scratch_game.make_move(action)  # Update scratch game
            self.latest_sim_path.append(node)  # Add this moment to the sim path

        # At the leaf node, save the available moves
        self.latest_valid_actions_mask = scratch_game.get_valid_moves()

        # Make the model call, search will resume from the next step() call
        return scratch_game.get_canonical_state()

    def select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        _, action, child = max(
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * cast(float, child.probability)
        value_score = float(child.value())
        return prior_score + value_score
