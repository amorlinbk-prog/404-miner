import os
import random
import numpy as np
import torch


def secure_randint(low: int, high: int) -> int:
    """ Return a random integer in [low, high] using os.urandom. """
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2**(8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, 'big')
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)


def set_random_seed(seed: int) -> None:
    """ Function for setting global seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False