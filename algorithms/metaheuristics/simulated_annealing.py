import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Tuple

import numpy as np

from algorithms.perturbations.diversification import random_truck_paired_select as rps
from algorithms.perturbations.diversification import (
    add_parking,
    fuse_trucks,
    group_parkings,
    shuffle_route,
    swap_interruta_random,
    transfer_node_random,
)
from algorithms.perturbations.intensification import (
    launch_drone_savings,
    or_opt_improvement,
    swap_interruta_savings,
    swap_truck_drone_savings,
    transfer_node_savings,
)
from models.solution import Solution
from models.vehicles import Truck
from utils.metrics import RouteCache
from utils.timer import FunctionTimer

sa_timer = FunctionTimer()

@dataclass
class ObjectiveTracker:
    """Utility for tracking objective function improvements."""
    def __init__(self):
        self.objective_delta: Dict[str, List[float]] = defaultdict(list)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for func_name, deltas in self.objective_delta.items():
            stats[func_name] = {
                'avg_delta': sum(deltas) / len(deltas),
                'worst_delta': min(deltas),
                'best_delta': max(deltas),
                'total_calls': len(deltas),
            }
        return stats

@sa_timer
def simulated_annealing(
    parameters: Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, RouteCache, RouteCache],
    initial_solution: Solution,
    n_iter: int = 200,
    initial_temperature: float = 1000.0,
    cooling_rate: float = 0.99,
    exclude_operators: List[str] = [],
) -> Tuple[Dict[Solution, Tuple[float, Dict[Truck, float]]], ObjectiveTracker]:
    """
    Performs Simulated Annealing optimization for the routing problem.

    Args:
        parameters: Problem instance parameters.
        initial_solution: Initial solution to start the search from.
        n_iter: Number of iterations.
        initial_temperature: Initial temperature for the annealing process.
        cooling_rate: Rate at which the temperature decreases.
        exclude_operators: List of perturbation operators to exclude.

    Returns:
        Dictionary mapping solutions to their emissions and makespan values.
    """
    mapper, truck_dm, drone_dm, times_truck, times_drone, truck_cache, drone_cache = parameters

    def calculate_emissions(sol: Solution) -> float:
        """Calculate the emissions for a given solution."""
        return sol.emissions(mapper, truck_dm, drone_dm)

    # Initialize solution memory with the initial solution
    solution_memory = {
        initial_solution: (
            calculate_emissions(initial_solution),
            initial_solution.makespan_per_truck(mapper, times_truck, times_drone),
        )
    }

    current_solution = initial_solution
    current_emissions = calculate_emissions(current_solution)
    best_solution = current_solution
    best_emissions = current_emissions

    # Prepare perturbation boxes
    def wrap_route_improvement(improvement_func):
        """Wraps route improvement functions to update the route in-place."""
        @wraps(improvement_func)
        def wrapper(route, *args):
            improved_route = improvement_func(route, *args)
            if improved_route is not route:
                route[:] = improved_route[:]  # Modify the list in-place
        return wrapper

    def get_perturbation_boxes(solution: Solution, truck_a: Truck, truck_b=None):
        """Generates a list of perturbation operations (boxes) to be applied."""
        single_box = [
            (group_parkings, (truck_a, mapper, truck_dm, drone_dm)),
            (swap_interruta_savings, (solution.trucks, mapper, truck_dm, truck_cache)),
            (transfer_node_savings, (solution.trucks, mapper, truck_dm, truck_cache)),
            (swap_truck_drone_savings, (truck_a, mapper, truck_dm, drone_dm, truck_cache, drone_cache)),
            (wrap_route_improvement(or_opt_improvement), (truck_a.route, truck_dm, mapper, truck_cache)),
            (shuffle_route, (truck_a,)),
            (launch_drone_savings, (truck_a, solution.drones, mapper, truck_dm, drone_dm, truck_cache, drone_cache)),
            (add_parking, (truck_a, solution.nodes)),
        ]

        if truck_b:
            paired_box = [
                (transfer_node_random, (truck_a, truck_b)),
                (fuse_trucks, (truck_a, truck_b)),
                (swap_interruta_random, (truck_a, truck_b)),
            ]
            full_box = single_box + paired_box
        else:
            full_box = single_box

        if exclude_operators:
            full_box = [op for op in full_box if op[0].__name__ not in exclude_operators]
        return full_box

    # Track objective improvements
    sa_tracker = ObjectiveTracker()

    # Main Simulated Annealing loop
    temperature = initial_temperature
    for i in range(1, n_iter + 1):
        # Generate a neighbor solution
        neighbor = deepcopy(current_solution)
        paired_trucks = rps(neighbor.trucks)

        # Get perturbation options
        if len(paired_trucks) > 1:
            func_box = get_perturbation_boxes(neighbor, paired_trucks[0], paired_trucks[1])
        else:
            func_box = get_perturbation_boxes(neighbor, paired_trucks[0])

        # Apply perturbation
        selected_function, args = random.choice(func_box)
        selected_function(*args)
        for truck in neighbor.trucks:
            truck.drop_unused_parkings()

        neighbor.last_improvement = selected_function.__name__
        neighbor_emissions = calculate_emissions(neighbor)

        # Calculate the change in emissions
        delta_emissions = neighbor_emissions - current_emissions

        # Accept the neighbor solution based on the acceptance probability
        if delta_emissions < 0 or random.random() < np.exp(-delta_emissions / temperature):
            current_solution = neighbor
            current_emissions = neighbor_emissions
            sa_tracker.objective_delta[selected_function.__name__].append(delta_emissions)

            # Update the best solution if necessary
            if current_emissions < best_emissions:
                best_solution = current_solution
                best_emissions = current_emissions

        # Cool down the temperature
        temperature *= cooling_rate

        # Store the current solution in memory
        solution_memory[current_solution] = (
            current_emissions,
            current_solution.makespan_per_truck(mapper, times_truck, times_drone),
        )

    return solution_memory, sa_tracker