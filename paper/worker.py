"""
Juggles many MCTS games in parallel, to avoid bottlenecking on
model calls on the GPU. Should be run in a separate process
to escape the GIL.
"""

from multiprocessing import Queue
from config import AlphaZeroConfig
from mcts import MCTS
from typing import Tuple, List
import time
import numpy as np
import numpy.typing as npt


def continually_run_mcts(requests: Queue, responses: Queue, finished_games: Queue):
    """
    Runs MCTS games continually, state machine style.
    """
    # Load config
    config = AlphaZeroConfig()

    # The number of games to process state machine like
    num_games = config.games_per_batch * config.game_batches_per_core

    pending_inference_requests: List[Tuple[int, npt.NDArray[np.float32]] | None] = [
        None
    ] * num_games  # List of games awaiting inference
    pending_release_games = []  # List of games that are over and will go to the window
    local_responses: List[Tuple[npt.NDArray[np.float32], np.float32] | None] = [
        None
    ] * num_games  # Local list to hold responses

    # Load up the games
    games = [MCTS()] * num_games  # List of MCTS games
    for i in range(num_games):
        game = MCTS()
        games[i] = game
        pending_inference_requests[i] = (
            i,
            game.first_call,
        )  # Initialize with the first call
        # For each batch request inferences
        if (i + 1) % config.games_per_batch == 0:
            requests.put(
                pending_inference_requests[(i + 1) - config.games_per_batch : (i + 1)]
            )

    # Begin infinite play loop
    while True:
        for i in range(num_games):
            game: MCTS = games[i]

            # Wait for a response if needed
            while local_responses[i] is None:
                if not responses.empty():
                    # Drain a batch of responses
                    response = responses.get()
                    for idx, prior, value in response:
                        local_responses[int(idx)] = (prior, value)
                else:
                    # Avoid busy waiting
                    time.sleep(0.01)

            prior, value = local_responses[i]  # type: ignore
            local_responses[i] = None  # Reset for next iteration

            # Advance MCTS
            request, result = game.step(prior, value)
            if request is not None:
                print("request", flush=True)
                pending_inference_requests[i] = (i, request)
            elif result is not None:
                pending_release_games.append(result)
                games[i] = MCTS()  # Game ended, no more steps needed
                pending_inference_requests[i] = (i, games[i].first_call)
                print("game ended")

            if (i + 1) % config.games_per_batch == 0:
                # If we have a full batch of requests, send them
                requests.put(
                    pending_inference_requests[
                        (i + 1) - config.games_per_batch : (i + 1)
                    ]
                )
                requests.empty()  # Flush buffer

        finished_games.put(pending_release_games)
        pending_release_games.clear()
