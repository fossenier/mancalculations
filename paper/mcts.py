"""
Monte Carlo Tree Search (MCTS) implemented for Mancala (Kalah) based on the
AlphaZero paper. Written to be multi-threaded across soarserver's 64 threads,
and to pass the work to the workers handling 4x A100 interaction. Not currently
setup to support virtual loss.
"""

from multiprocessing import Queue
import numpy as np
from inference_batcher import inference_queue
from config import AlphaZeroConfig
from kalah import KalahGame
from numpy.typing import NDArray
from typing import Tuple


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

    def expand(self, prior: NDArray[np.float32], valid_actions: list[int]) -> None:
        """
        Expands the node by adding children for each valid action with a prior probability.
        This is called when the node is a leaf node and needs to be expanded.
        """
        if self.expanded():
            raise ValueError("Cannot expand an already expanded node.")

        # valid_actions is a mask (e.g., [1, 0, 1, 0, 1, 0]) over actions [0-5]
        for action, is_valid in enumerate(valid_actions):
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
    def __init__(self, worker: int, requests: Queue, window: Queue) -> None:
        """
        Sets up all variables needed to run the MCTS search, and also puts in
        a model request for the base root state to get the queue started.
        """
        # To make model calls, and save the game when complete
        self.requests: Queue = requests
        self.window: Queue = window
        self.worker: int = worker  # Worker ID for queue sync

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
        self.latest_valid_actions: list[int] = []

        # Prompt for  the first model response, which will initialize the root on first step() call
        requests.put((self.worker, self.game.get_canonical_state()))

    def step(self, prior: NDArray[np.float32], value: np.float32) -> bool:
        """
        Returns True if the game is still running, False if it has ended.

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
                self.window.put(self.chosen_path)
                return False  # Game is over
            else:
                return True  # Game continues (new root node will be created next time)

        # Otherwise, run another simulation for this root node
        self.simulate()
        return True

    def backpropagate(
        self, prior: NDArray[np.float32], value: np.float32, player: int
    ) -> None:
        """
        Backpropogates the model's value prediction up the latest visited path
        (and updates visit counts + values too).
        """
        # No nodes in the latest chosen path implies this will be a root node
        if not self.latest_sim_path:
            self.root = MCTSNode(action=None, player=player, prior=None, parent=None)
            self.root.visit_count = 1
            self.root.total_value = value
            self.latest_sim_path.append(self.root)
        # Backpropagate the value up the tree
        else:
            # First, expand the leaf node
            # NOTE: This shouldn't attempt to expand the terminal state, due to the
            # control flow in step(), but it will be safe and produce no children
            # if that happens somehow (due to expand()'s use of get_valid_moves())
            leaf_node = self.latest_sim_path[-1]
            leaf_node.expand(prior, self.latest_valid_actions)

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
            action = np.random.choice(
                [action for action, _ in probabilities],
                p=[prob for _, prob in probabilities],
            )
        # Be deterministic, choose the child that was most visited
        else:
            action = np.argmax(
                [child.visit_count for child in self.root.children.values()]
            )

        # Update the game state
        self.game.make_move(int(action))

    def prepare_statistics(self) -> None:
        # TODO: prepare self.chosen_path for the window
        pass

    def simulate(self) -> None:
        # TODO: Go to leaf node, and model call
        # Update latest_valid_actions
        pass


import numpy as np


def select_action(visit_counts, temperature=1.0):
    """
    Select action based on visit counts and temperature.

    Args:
        visit_counts: array of visit counts N(s,a) for each action
        temperature: temperature parameter τ

    Returns:
        selected action index
    """
    if temperature == 0:
        # When temperature is 0, select action with highest visit count
        return np.argmax(visit_counts)
    else:
        # Apply temperature scaling
        scaled_counts = np.power(visit_counts, 1.0 / temperature)
        # Normalize to get probabilities
        probabilities = scaled_counts / np.sum(scaled_counts)
        # Sample action according to probabilities
        return np.random.choice(len(visit_counts), p=probabilities)


def mcts_action_selection(visit_counts, move_number):
    """
    Select action with temperature schedule: τ=1 for first 30 moves, τ=0 afterwards
    """
    temperature = 1.0 if move_number < 30 else 0.0
    return select_action(visit_counts, temperature)

    def softmax_sample(
        visit_count: int, action_visits: list[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Samples an action based on the softmax distribution of visit counts.
        """
        # Exploration rate C(s)
        # C(s) = log((1 + N(s) + pb_c_base) / pb_c_base) + pb_c_init
