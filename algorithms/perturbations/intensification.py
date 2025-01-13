"""
Intensification algorithms for improving routes in the LRP-D
"""
from typing import Dict, List

import numpy as np

from models.nodes import Node
from utils.metrics import RouteCache, calculate_route_metric
from algorithms.perturbations.decorators import perturbations_timer

@perturbations_timer
def two_opt_improvement(route: List[Node], distance_matrix: np.ndarray, mapper: Dict[str, int], cache: RouteCache) -> List[Node]:
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

def single_or_opt_improvement(route: List[Node], distance_matrix: np.ndarray, mapper: Dict[str, int], cache: RouteCache, string_length: int = 3) -> List[Node]:
    """
    Performs a single OR-opt improvement on the route if possible.
    OR-opt removes a string of consecutive nodes and reinserts them at a different position.
    
    Args:
        route: List of nodes representing the current route
        distance_matrix: Matrix containing distances between nodes
        mapper: Dictionary mapping node names to matrix indices
        cache: Cache object for storing route calculations
        string_length: Length of the consecutive nodes to move (default: 3)
        
    Returns:
        The improved route or the original route if no improvement is found
    """
    current_distance = calculate_route_metric(route, distance_matrix, mapper, cache)
    n = len(route)
    
    # Don't allow string_length longer than route/2
    string_length = min(string_length, n // 2)
    
    # Try all possible string removals
    for i in range(1, n - string_length):
        # Get the string of nodes to be moved
        string_to_move = route[i:i + string_length]
        
        # Try inserting the string at all possible positions
        for j in range(1, n - string_length):
            # Skip if insertion point is within or adjacent to removal area
            if j >= i - 1 and j <= i + string_length:
                continue
                
            # Create new route by removing string and inserting at new position
            new_route = route[:i] + route[i + string_length:]  # Remove string
            new_route = new_route[:j] + string_to_move + new_route[j:]  # Insert string
            
            # Calculate new distance
            new_distance = calculate_route_metric(new_route, distance_matrix, mapper, cache)
            
            # If improvement found, return the new route
            if new_distance < current_distance:
                return new_route
    
    # Return original route if no improvement found
    return route

@perturbations_timer
def or_opt_improvement(route: List[Node], distance_matrix: np.ndarray, mapper: Dict[str, int], cache: RouteCache) -> List[Node]:
    """
    Performs OR-opt improvements using different string lengths until no improvement is found.
    
    Args:
        route: List of nodes representing the current route
        distance_matrix: Matrix containing distances between nodes
        mapper: Dictionary mapping node names to matrix indices
        cache: Cache object for storing route calculations
        
    Returns:
        The improved route after all possible OR-opt moves
    """
    improved = True
    current_route = route
    
    while improved:
        improved = False
        # Try different string lengths (1 to 3 is common in practice)
        for string_length in range(1, 4):
            new_route = single_or_opt_improvement(current_route, distance_matrix, mapper, cache, string_length)
            if new_route != current_route:
                current_route = new_route
                improved = True
                break
    
    return current_route