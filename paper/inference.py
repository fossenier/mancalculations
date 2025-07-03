"""
The model calls that will return policy / value predicitons
"""

from typing import Tuple
import numpy.typing as npt
import numpy as np


# TODO: eventually this will be rewritten to accept stacked vectors or
# whatever for batch processing
def inference_call() -> Tuple[npt.NDArray[np.float32], np.float32]:
    policy = np.full((6,), 1 / 6, dtype=np.float32)
    value = np.float32(0.2)
    return policy, value
