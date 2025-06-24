import queue

from threading import Event


inference_queue = queue.Queue()

class InferenceJob:
    def __init__(self, state):
        self.state = state
        self.result = None
        self.ready = Event()
