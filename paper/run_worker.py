from worker import continually_run_mcts
from window import Window
from multiprocessing import Queue, Process
from inference import inference_call
from typing import List, Tuple
import numpy.typing as npt
import numpy as np


def run_window(games):
    """Wrapper function to run Window in a separate process"""
    window = Window(games)
    window.run()  # This will run forever


def main():
    requests = Queue()
    responses = Queue()
    games = Queue()

    # Create processes to run on separate cores
    mcts_process = Process(
        target=continually_run_mcts, args=(requests, responses, games)
    )
    window_process = Process(target=run_window, args=(games,))

    # You'll also need an inference process to handle requests/responses
    # inference_process = Process(target=your_inference_function, args=(requests, responses))

    # Start both processes
    mcts_process.start()
    window_process.start()
    # inference_process.start() 

    while True:
        try:
            # TODO the first value is the worker idx, pass back the response to the
            # correct queue
            request: List[Tuple[int, npt.NDArray[np.float32]]] = requests.get(
                timeout=0.1
            )
            response = [(idx, inference_call()) for idx, _ in request]

            responses.put(response)
        except Exception:
            continue

    # Keep the main process alive and wait for them
    try:
        mcts_process.join()
        window_process.join()
        # inference_process.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        mcts_process.terminate()
        window_process.terminate()
        # inference_process.terminate()
        mcts_process.join()
        window_process.join()
        # inference_process.join()


if __name__ == "__main__":
    main()
