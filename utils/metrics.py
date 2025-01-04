from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from models.nodes import Nodes


@dataclass
class RouteCache:
    """Cache for storing route calculations."""
    metric: Dict[tuple, float] = None
    
    def __post_init__(self):
        self.metric = {}

def get_index_mapper(matrix: pd.DataFrame) -> Dict[str, int]:
    return {index: idx for idx, index in enumerate(matrix.index)}

def calculate_route_metric(
    route: List[Nodes],
    numpy_matrix: np.ndarray,
    index_map: dict,
    cache: RouteCache
) -> float:
    """
    Cached calculation of route metrics (emissions or time). Optimized.
    Takes precomputed numpy matrix and index map as arguments.
    """
    route_key = tuple(node.id for node in route)
    if route_key in cache.metric:
        return cache.metric[route_key]

    # Fetch indices corresponding to node IDs
    indices = [index_map[node.id] for node in route]
    # Calculate the metric
    metric = np.sum(numpy_matrix[indices[i], indices[i + 1]] for i in range(len(indices) - 1))

    # Cache the result
    cache.metric[route_key] = metric
    return metric