import numpy as np
import cProfile
import pstats
import io
from multiprocessing import get_context, set_start_method, Event
from paper.worker import continually_run_mcts


def profile_wrapper(func, *args, **kwargs):
    """Wrapper to profile a function in a subprocess"""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        func(*args, **kwargs)
    finally:
        profiler.disable()

        # Save stats to a file with process ID
        import os

        filename = f"profile_{func.__name__}_{os.getpid()}.stats"
        profiler.dump_stats(filename)

        # Also print summary
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions
        print(f"\n=== Profile for {func.__name__} (PID: {os.getpid()}) ===")
        print(s.getvalue())


def consumer_with_profiling(requests, responses, shutdown_event):
    """Wrapper that profiles the consumer function"""

    def consumer():
        while not shutdown_event.is_set():
            if not requests.empty():
                request = requests.get()
                response = [
                    (request[i][0], np.full(6, 0.2, dtype=np.float32), 0.5)
                    for i in range(len(request))
                ]
                responses.put(response)
                responses.empty()
            else:
                # Small sleep to prevent busy waiting
                shutdown_event.wait(0.01)

    profile_wrapper(consumer)


def mcts_with_profiling(requests, responses, window, shutdown_event):
    """Wrapper that profiles the MCTS function"""

    # Modify continually_run_mcts to accept shutdown_event or wrap it
    def mcts_wrapper():
        # You'll need to modify continually_run_mcts to check shutdown_event
        # For now, let's simulate it
        import time

        start_time = time.time()
        while not shutdown_event.is_set() and (time.time() - start_time) < 20:
            # Call the actual MCTS function in chunks or modify it to check shutdown
            # This is a placeholder - you'll need to adapt based on your MCTS implementation
            time.sleep(0.1)

    profile_wrapper(mcts_wrapper)


def test_continually_run_mcts():
    ctx = get_context("fork")
    requests = ctx.Queue()
    responses = ctx.Queue()
    window = ctx.Queue()
    shutdown_event = ctx.Event()

    # Start processes with profiling wrappers
    mcts_process = ctx.Process(
        target=mcts_with_profiling, args=(requests, responses, window, shutdown_event)
    )
    mcts_process.start()

    consumer_process = ctx.Process(
        target=consumer_with_profiling, args=(requests, responses, shutdown_event)
    )
    consumer_process.start()

    import time

    time.sleep(20)

    # Signal shutdown
    print("Signaling shutdown...")
    shutdown_event.set()

    # Give processes time to finish cleanly
    mcts_process.join(timeout=5)
    consumer_process.join(timeout=5)

    # Only terminate if they didn't finish cleanly
    if mcts_process.is_alive():
        print("Force terminating MCTS process")
        mcts_process.terminate()
        mcts_process.join()

    if consumer_process.is_alive():
        print("Force terminating consumer process")
        consumer_process.terminate()
        consumer_process.join()


if __name__ == "__main__":
    set_start_method("fork", force=True)
    test_continually_run_mcts()
    print("Test completed successfully.")
    print("\nProfile files created:")
    import glob

    for f in glob.glob("profile_*.stats"):
        print(f"  - {f}")
