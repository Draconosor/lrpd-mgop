from models.nodes import Nodes
from typing import List, Dict
from utils.metrics import RouteCache, calculate_route_metric
import numpy as np

def single_2opt_improvement(route: List[Nodes], distance_matrix: np.ndarray, mapper: Dict[str, int], cache: RouteCache) -> List[Nodes]:
    """
    Performs a single 2-opt improvement on the route if possible.
    Returns the improved route or the original route if no improvement is found.
    """
    current_distance = calculate_route_metric(route, distance_matrix, mapper, cache)
    
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_distance = calculate_route_metric(new_route, distance_matrix, mapper, cache)
            
            if new_distance < current_distance:
                return new_route
                
    return route

def twoopt_until_local_optimum(route: List[Nodes], distance_matrix: np.ndarray, mapper: Dict[str, int], cache: RouteCache) -> List[Nodes]:
    """
    Applies 2-opt improvements repeatedly until no further improvements can be made.
    """
    cambio = 1
    count = 0
    while cambio != 0:
        count += 1
        inicial = calculate_route_metric(route, distance_matrix, mapper, cache)
        sol = single_2opt_improvement(route, distance_matrix, mapper, cache)
        final = calculate_route_metric(sol, distance_matrix, mapper, cache)
        anti = cambio
        cambio = np.abs(final-inicial)
        if anti == cambio or cambio < 0.0005 or count > 50:
            cambio = 0
            
    return route