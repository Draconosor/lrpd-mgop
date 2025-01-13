import random
import numpy as np

def global_seed_setter(seed: int):
    """
    Set the global random seed for reproducibility.

    This function sets the seed for both NumPy and the built-in random module,
    ensuring that random number generation is consistent across runs.

    Parameters:
    seed (int): The seed value to set for the random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)