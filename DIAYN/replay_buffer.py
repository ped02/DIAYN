import random
from collections import deque
from typing import Optional, List, Callable


class ReplayBuffer:
    def __init__(
        self, buffer_size, post_processor: Callable, seed: Optional[int] = None
    ):
        self.buffer = deque(maxlen=buffer_size)

        self.post_processor = post_processor

        self.seed = seed
        self.random_state = random.Random(self.seed)  # nosec

    def add(self, item, split_first_dim=False):
        if split_first_dim:
            self.buffer.extend(list(zip(*item)))
        else:
            self.buffer.append(item)

    def sample(self, size: int = 1) -> List:
        samples = self.random_state.sample(self.buffer, size)
        if self.post_processor is not None:
            samples = self.post_processor(samples)
        return samples

    def get_rng_state(self):
        return self.random_state.getstate()

    def set_rng_state(self, rng_state):
        self.random_state.setstate(rng_state)
