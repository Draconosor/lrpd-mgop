import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from algorithms.perturbations import (close_parking_lot_random, group_parkings,
                                      open_truck, optimize_drone_assignments,
                                      random_truck_paired_select,
                                      shuffle_route, swap_interruta_random,
                                      swap_interruta_saving,
                                      transfer_node_random)
from models.solution import Solution
from models.vehicles import Truck
from utils.metrics import RouteCache
from utils.timer import FunctionTimer

ls_timer = FunctionTimer()

@ls_timer
def local_search(parameters: Tuple[Dict[str,int],np.ndarray, np.ndarray, np.ndarray, np.ndarray], initial_solution: Solution, n_iter: int = 200, n_size: int = 20) -> Dict[Solution, Tuple[float, Dict[Truck, float]]]:
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
    def get_perturbation_boxes(solution: Solution, truck_a: Truck, truck_b=None):
        single_box = [
            (group_parkings, (truck_a, mapper, drone_dm, truck_dm, truck_cache)),
            (close_parking_lot_random, (solution.trucks, mapper, truck_dm, drone_dm, truck_cache)),
            (shuffle_route, (truck_a, mapper, truck_dm, truck_cache)),
            (optimize_drone_assignments, (solution, mapper, truck_dm, drone_dm, truck_cache, drone_cache)),
            (swap_interruta_saving, (solution.trucks, mapper, truck_dm, truck_cache)),
            (open_truck, (solution.trucks, mapper, truck_dm, truck_cache))
        ]
        
        if truck_b:
            paired_box = [
                (transfer_node_random, (truck_a, truck_b, mapper, truck_dm, truck_cache)),
                (swap_interruta_random, (truck_a, truck_b, mapper, truck_dm, truck_cache))
            ]
            return single_box + paired_box
        return single_box

    # Main iteration loop
    for i in range(1, n_iter + 1):
        neighbors: List[Solution] = []
        
        # Generate neighbors
        for _ in range(n_size):
            neighbor = deepcopy(start_sol)
            paired_trucks = random_truck_paired_select(neighbor.trucks)
            
            # Get perturbation options
            if len(paired_trucks) > 1:
                func_box = get_perturbation_boxes(neighbor, paired_trucks[0], paired_trucks[1])
            else:
                func_box = get_perturbation_boxes(neighbor, paired_trucks[0])
            
            # Apply perturbation and optimization
            selected_function, args = random.choice(func_box)
            selected_function(*args)
            
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
            start_sol = n
            solution_memory[n] = (n_emissions, n.makespan_per_truck(mapper, times_truck, times_drone))
            valid_neighbor_found = True
            break

        if not valid_neighbor_found:
            # Handle the case where all neighbors are already in solution_memory
            # Example: Pick the neighbor with the least emissions
            best_neighbor = neighbors[0]  # Neighbors are already sorted by emissions
            best_neighbor.id = i
            start_sol = best_neighbor
            solution_memory[best_neighbor] = (
                calculate_emissions(best_neighbor),
                best_neighbor.makespan_per_truck(mapper, times_truck, times_drone)
            )

                        

    return solution_memory