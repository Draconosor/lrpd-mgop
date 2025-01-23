"""
This module contains the implementation of the local search algorithm for the routing problem.
"""


import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Tuple

import numpy as np

from algorithms.perturbations.diversification import random_truck_paired_select as rps

from algorithms.perturbations.diversification import (add_parking, 
                                                      fuse_trucks, 
                                                      group_parkings,
                                                      shuffle_route, 
                                                      swap_interruta_random, 
                                                      transfer_node_random)
from algorithms.perturbations.intensification import (launch_drone_savings,
                                                      or_opt_improvement,
                                                      swap_interruta_savings,
                                                      swap_truck_drone_savings,
                                                      transfer_node_savings,
                                                      two_opt_improvement)
from models.solution import Solution
from models.vehicles import Truck
from utils.metrics import RouteCache
from utils.timer import FunctionTimer

ls_timer = FunctionTimer()

@dataclass
class ObjectiveTracker:
    """Utility for timing function executions."""
    def __init__(self):
        self.objective_delta: Dict[str, List[float]] = defaultdict(list)
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for func_name, deltas in self.objective_delta.items():
            stats[func_name] = {
                'avg_delta': sum(deltas) / len(deltas),
                'worst_delta': min(deltas),
                'best_delta': max(deltas),
                'total_calls': len(deltas)
            }
        return stats

@ls_timer
def local_search(parameters: Tuple[Dict[str,int],np.ndarray, np.ndarray, np.ndarray, np.ndarray], initial_solution: Solution, n_iter: int = 200, n_size: int = 10) -> Tuple[Dict[Solution, Tuple[float, Dict[Truck, float]]], ObjectiveTracker]:
    """
    Performs local search optimization for the routing problem.
    
    Args:
        instance_id: Identifier for the problem instance
        n_iter: Number of iterations
        n_size: Number of neighbors to generate per iteration
    
    Returns:
        Dictionary mapping solutions to their emissions and makespan values
    """
    # Read instance and initialize
    
    mapper, truck_dm, drone_dm, times_truck, times_drone = parameters
    
    
    
    def calculate_emissions(sol: Solution):
        """
        Calculate the emissions for a given solution.

        Args:
            sol (Solution): The solution object containing the necessary data to calculate emissions.

        Returns:
            float: The total emissions calculated based on the provided solution.
        """
        return sol.emissions(mapper, truck_dm, drone_dm)
    
    # Initialize solution memory with initial solution
    solution_memory = {initial_solution: (
        calculate_emissions(initial_solution),
        initial_solution.makespan_per_truck(mapper, times_truck, times_drone)
    )}
    
    start_sol = initial_solution
    truck_cache = RouteCache()  # Create once and reuse
    drone_cache = RouteCache()  
    
    # Prepare perturbation boxes outside the loop
    def wrap_route_improvement(improvement_func):
        """
        Wraps route improvement functions to update the route reference with the function output.
        The route argument will be a reference to truck_a.route
        """
        @wraps(improvement_func)
        def wrapper(route, *args):
            improved_route = improvement_func(route, *args)
            if improved_route is not route:
                route[:] = improved_route[:]  # Modify the list in-place
        return wrapper

    def get_perturbation_boxes(solution: Solution, truck_a: Truck, truck_b=None):
        """
        Generates a list of perturbation operations (boxes) to be applied to the given solution.
        Parameters:
        solution (Solution): The current solution containing the trucks and their routes.
        truck_a (Truck): The primary truck to which perturbations will be applied.
        truck_b (Truck, optional): An optional secondary truck for paired perturbations. Defaults to None.
        Returns:
        list: A list of tuples where each tuple contains a perturbation function and its corresponding arguments.
        """
        single_box = [
            (group_parkings, (truck_a, mapper, truck_dm, drone_dm)),
            (swap_interruta_savings, (solution.trucks, mapper, truck_dm, truck_cache)),
            (transfer_node_savings, (solution.trucks, mapper, truck_dm, truck_cache)),
            (swap_truck_drone_savings, (truck_a, mapper, truck_dm, drone_dm, truck_cache, drone_cache)),
            (wrap_route_improvement(two_opt_improvement), (truck_a.route, truck_dm, mapper, truck_cache)),
            (wrap_route_improvement(or_opt_improvement), (truck_a.route, truck_dm, mapper, truck_cache)),
            (shuffle_route, (truck_a,)),
            (launch_drone_savings, (truck_a, solution.drones, mapper, truck_dm, drone_dm, truck_cache, drone_cache)),
            (add_parking, (truck_a, solution.nodes))
        ]
        
        if truck_b:
            paired_box = [
                (transfer_node_random, (truck_a, truck_b)),
                (fuse_trucks, (truck_a, truck_b)),
                (swap_interruta_random, (truck_a, truck_b))
            ]
            return single_box + paired_box
        return single_box
    # Track objective improvements
    
    sm_tracker = ObjectiveTracker()
    p_tracker = ObjectiveTracker()
    
    # Main iteration loop
    for i in range(1, n_iter + 1):
        neighbors: List[Solution] = []
        
        # Generate neighbors
        for _ in range(n_size):
            neighbor = deepcopy(start_sol)
            paired_trucks = rps(neighbor.trucks)
            
            # Get perturbation options
            if len(paired_trucks) > 1:
                func_box = get_perturbation_boxes(neighbor, paired_trucks[0], paired_trucks[1])
            else:
                func_box = get_perturbation_boxes(neighbor, paired_trucks[0])
            
            # Apply perturbation and optimization
            #print(calculate_emissions(neighbor))
            selected_function, args = random.choice(func_box)
            selected_function(*args)
            for truck in neighbor.trucks:
                truck.drop_unused_parkings()
            #print(calculate_emissions(neighbor))
            neighbor.last_improvement = selected_function.__name__
            p_tracker.objective_delta[selected_function.__name__].append(calculate_emissions(start_sol) - calculate_emissions(neighbor))
            
            neighbors.append(neighbor)

            
        # Sort neighbors by objective
        neighbors.sort(key = lambda x: calculate_emissions(x))
        # Check if any valid neighbor exists
        valid_neighbor_found = False

        for n in neighbors:
            n_emissions = calculate_emissions(n)
            if any(n_emissions == v[0] for v in solution_memory.values()):
                continue
            # Valid neighbor found
            n.id = i
            sm_tracker.objective_delta[n.last_improvement].append(calculate_emissions(start_sol) - calculate_emissions(n))
            start_sol = n
            solution_memory[n] = (n_emissions, n.makespan_per_truck(mapper, times_truck, times_drone))
            valid_neighbor_found = True
            break

        if not valid_neighbor_found:
            # Handle the case where all neighbors are already in solution_memory
            # Example: Pick the neighbor with the least emissions
            best_neighbor = neighbors[0]  # Neighbors are already sorted by emissions
            best_neighbor.id = i
            sm_tracker.objective_delta[best_neighbor.last_improvement].append(calculate_emissions(start_sol) - calculate_emissions(best_neighbor))
            start_sol = best_neighbor
            solution_memory[best_neighbor] = (
                calculate_emissions(best_neighbor),
                best_neighbor.makespan_per_truck(mapper, times_truck, times_drone)
            )

    return solution_memory, sm_tracker