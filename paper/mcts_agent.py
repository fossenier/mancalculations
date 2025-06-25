"""
Juggles many MCTS games in parallel, to avoid bottlenecking on
model calls on the GPU. Should be run in a separate process
to escape the GIL.
"""

from multiprocessing import Queue
from config import AlphaZeroConfig
from kalah import KalahGame

def continually_run_mcts(requests: Queue, responses: Queue, window: Queue):
    """
    Runs MCTS games continually, state machine style.
    """
    # Load config
    config = AlphaZeroConfig()
    
    # The number of games to process state machine like
    num_games = config.games_per_core
    
    games = [KalahGame() for _ in range(num_games)]
    
    while True:
        for i in range(num_games):
            game = games[i]
            # check if game is over
                # push to window, make new game
            
            # check if game is awaiting
                # check inference queue
                
            # game should now make a model request+
        
    