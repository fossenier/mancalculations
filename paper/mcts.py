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
        self, action: int, player: int, prior: float, parent: "MCTSNode"
    ) -> None:
        self.action: int = action  # Pit moved from to reach this state
        self.visit_count: int = 0  # N = visit count
        self.total_value: float = 0.0  # W = total value
        # Q = W/N = average value = state value of the node
        self.probability: float = prior  # P = probability of taking this action
        # (as determined by the policy network when given the parent state)
        self.player: int = player  # Player who made the move to reach this state
        self.children: dict[int, "MCTSNode"] = {}  # Child nodes, keyed by action
        self.parent: "MCTSNode | None" = parent  # Parent node, None for root node

    def expanded(self) -> bool:
        """
        Returns True if the node has been expanded (i.e., has children).
        """
        return len(self.children) > 0

    def value(self) -> float:
        """
        Returns the value of the node, which is the average value.
        """
        if self.visit_count == 0:
            return 0.0
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
        # Current simulation count, 800 inititally to force root creation
        self.current_simulation: int = self.max_simulations
        # The current MCTS root node, initially None
        self.root: MCTSNode | None = None
        # The real actions taken during the course of the game
        self.chosen_path: list[MCTSStatistic] = []
        # The latest simulation to leaf node, to backpropogate
        self.latest_sim_path: list[MCTSNode] = []

        # Prompt for  the first model response, which will initialize the root on first step() call
        requests.put((self.worker, self.game.get_canonical_state()))

    # Play game, BUT, play distributed game, one step in each of MAX_GAMES

    def step(self, prior: NDArray[np.float32], value: np.float32) -> bool:
        """
        Moves along MCTS behaviour, until a model call is needed, at which point
        it pushes that request to the requests queue. Resume the search once the
        model returns the call by passing the prior and value back to this function.
        """
        return True

    def search_once(self) -> NDArray[np.float32]:
        """
        Initializes
        """
        root = MCTSNode(action=-1, player=self.game.current_player())
        # Run the MCTS search
        self._run_mcts(root)
