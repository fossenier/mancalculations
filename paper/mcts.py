"""
Monte Carlo Tree Search (MCTS) implemented for Mancala (Kalah) based on the
AlphaZero paper. Written to be multi-threaded across soarserver's 64 threads,
and to pass the work to the workers handling 4x A100 interaction. Not currently
setup to support virtual loss.
"""
from inference_batcher import inference_queue


class MCTSNode:
    """
    Node for a MCTS tree.
    """

    def __init__(self, action: int) -> None:
        self.action: int = action # Pit moved from to reach this state
        self.visit_count: int = 0 # N = visit count
        self.total_reward: float = 0.0 # W = total reward
        # Q = W/N = average reward = state value of the node
        self.probability: float = 0.0 # P = probability of taking this action (as determined
                                      # by the policy network when given the parent state)