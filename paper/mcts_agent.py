"""
Juggles many MCTS games in parallel, to avoid bottlenecking on
model calls on the GPU. Should be run in a separate process
to escape the GIL.
"""

from multiprocessing import Queue

def continually_run_mcts(request_queue: Queue, response_queue: Queue):
    """
    Runs MCTS games continually, state machine style.
    """
    