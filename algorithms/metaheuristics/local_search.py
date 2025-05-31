"""
This module contains the implementation of the local search algorithm for the routing problem.
"""


import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Tuple
from weakref import ref

import numpy as np

from algorithms.perturbations.diversification import \
    random_truck_paired_select as rps
    
from algorithms.perturbations.diversification import (shuffle_route,
                                                      add_parking, 
                                                      fuse_trucks,
                                                      group_parkings,
                                                      swap_interruta_random,
                                                      transfer_node_random)
from algorithms.perturbations.intensification import (launch_drone_savings,
                                                      or_opt_improvement,
                                                      swap_interruta_savings,
                                                      swap_truck_drone_savings,
                                                      transfer_node_savings)
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
def local_search(multiobjective_params: Tuple[float,float, float], parameters: Tuple[Dict[str,int],np.ndarray, np.ndarray, np.ndarray, np.ndarray, RouteCache, RouteCache], 
                 initial_solution: Solution, n_iter: int = 200, n_size: int = 20, exclude_operators: List[str] = [], specify_operators: List[str] = []) -> Tuple[Dict[Solution, Dict], Tuple[ObjectiveTracker, ObjectiveTracker]]:
    """
    Performs local search optimization for the routing problem.
    
    Args:
        instance_id: Identifier for the problem instance
        n_iter: Number of iterations
        n_size: Number of neighbors to generate per iteration
    
    Returns:
        Dictionary mapping solutions to their emissions and makespan values and objectve trackers tuple
    """
    # Read instance and initialize
    
    mapper, truck_dm, drone_dm, times_truck, times_drone, truck_cache, drone_cache = parameters
    ref_em, ref_makespan, alfa = multiobjective_params
    
    def calculate_objective(sol: Solution, alfa: float = alfa):
        """
        Calculate the weighted objective for a given solution.

        Args:
            sol (Solution): The solution object containing the necessary data to calculate emissions.

        Returns:
            float: The total emissions calculated based on the provided solution.
        """
        return sol.emissions(mapper, truck_dm, drone_dm)*alfa + (1-alfa)*max(sol.makespan_per_truck(mapper, times_truck, times_drone).values())
    
    # Initialize solution memory with initial solution
    solution_memory = {
        initial_solution: {
            'objective' : calculate_objective(initial_solution),
            'emissions': initial_solution.emissions(mapper, truck_dm, drone_dm),
            'makespan_per_truck' : initial_solution.makespan_per_truck(mapper, times_truck, times_drone)
            }
        }
    
    start_sol = initial_solution
    
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
            full_box =  single_box + paired_box
        else:    
            full_box =  single_box
        
        if exclude_operators:
            full_box = [op for op in full_box 
                      if op[0].__name__ not in exclude_operators]
        if specify_operators:
            full_box = [op for op in full_box 
                      if op[0].__name__ in specify_operators]
        return full_box
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
            selected_function, args = random.choice(func_box)
            selected_function(*args)
            for truck in neighbor.trucks:
                truck.drop_unused_parkings()
            neighbor.last_improvement = selected_function.__name__
            p_tracker.objective_delta[selected_function.__name__].append(calculate_objective(start_sol) - calculate_objective(neighbor))
            neighbors.append(neighbor)


            
        # Sort neighbors by objective
        neighbors.sort(key = lambda x: calculate_objective(x))
        # Check if any valid neighbor exists
        valid_neighbor_found = False

        for n in neighbors:
            n_objective = calculate_objective(n)
            if any(n_objective == v['objective'] for v in solution_memory.values()):
                continue
            # Valid neighbor found
            n.id = i
            sm_tracker.objective_delta[n.last_improvement].append(calculate_objective(start_sol) - calculate_objective(n))
            start_sol = n
            solution_memory[n] = {
                'objective' : n_objective,
                'emissions': n.emissions(mapper, truck_dm, drone_dm),
                'makespan_per_truck' : n.makespan_per_truck(mapper, times_truck, times_drone)
                }
            valid_neighbor_found = True
            break

        if not valid_neighbor_found:
            # Handle the case where all neighbors are already in solution_memory
            # Example: Pick the neighbor with the least emissions
            best_neighbor = neighbors[0]  # Neighbors are already sorted by emissions
            best_neighbor.id = i
            sm_tracker.objective_delta[best_neighbor.last_improvement].append(calculate_objective(start_sol) - calculate_objective(best_neighbor))
            start_sol = best_neighbor
            solution_memory[best_neighbor] = {
                'objective' : calculate_objective(best_neighbor),
                'emissions': best_neighbor.emissions(mapper, truck_dm, drone_dm),
                'makespan_per_truck': best_neighbor.makespan_per_truck(mapper, times_truck, times_drone)
            }

    return solution_memory, (sm_tracker, p_tracker)