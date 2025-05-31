"""
Intensification algorithms for improving routes in the LRP-D
"""
from typing import Dict, List, Union

import numpy as np

from algorithms.perturbations.decorators import perturbations_timer
from models.nodes import Node
from models.vehicles import (Drone, Truck, add_node, launch_drone, remove_node,
                             swap_nodes, transfer_node)
from utils.metrics import RouteCache, calculate_route_metric


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
    failures = 0
    
    while improved and failures < 50:
        improved = False
        # Try different string lengths (1 to 3 is common in practice)
        for string_length in range(1, 4):
            new_route = single_or_opt_improvement(current_route, distance_matrix, mapper, cache, string_length)
            if new_route != current_route:
                current_route = new_route
                improved = True
                break
            failures += 1
    
    return current_route

@perturbations_timer
def swap_interruta_savings(trucks: List[Truck], mapper: Dict[str, int], matrix: np.ndarray, cache: RouteCache) -> None:
    """
    Attempts to improve the overall route efficiency by swapping nodes between 
    different truck routes based on potential savings.
    Args:
        trucks (List[Truck]): A list of Truck objects representing the fleet.
        mapper (Dict[str, int]): A dictionary mapping node identifiers to their indices in the distance matrix.
        matrix (np.ndarray): A 2D numpy array representing the distance matrix.
        cache (RouteCache): A cache object to store and retrieve precomputed route metrics.
    Returns:
        None: This function modifies the routes of the trucks in place.
    The function performs the following steps:
        1. Filters out unused trucks.
        2. Calculates potential savings for swapping nodes between different truck routes.
        3. Checks capacity constraints to ensure feasibility of the swap.
        4. Evaluates the new routes and calculates the savings.
        5. Sorts the potential savings and applies the best feasible swap to improve route efficiency.
    """
    used_trucks = [truck for truck in trucks if truck.is_used]
    if len(used_trucks) < 2:
        return

    savings = []
    
    # Calculate all potential savings
    for i, truck_a in enumerate(used_trucks[:-1]):
        for truck_b in used_trucks[i+1:]:
            for pos_a in range(1, len(truck_a.route) - 1):
                for pos_b in range(1, len(truck_b.route) - 1):
                    node_a = truck_a.route[pos_a]
                    node_b = truck_b.route[pos_b]
                    
                    # Calculate associated drone demands for both nodes if they're parking lots
                    drones_a = []
                    drones_b = []
                    demand_a = node_a._demand
                    demand_b = node_b._demand
                    
                    if node_a.node_type == 'Parking Lot':
                        drones_a = [d for d in truck_a.drones if d.route[0] == node_a]
                        demand_a += sum(d.weight for d in drones_a)
                        
                    if node_b.node_type == 'Parking Lot':
                        drones_b = [d for d in truck_b.drones if d.route[0] == node_b]
                        demand_b += sum(d.weight for d in drones_b)
                    
                    # Check capacity constraints for both trucks after swap
                    new_cap_a = truck_a.used_capacity - demand_a + demand_b
                    new_cap_b = truck_b.used_capacity - demand_b + demand_a
                    
                    if new_cap_a > truck_a.capacity or new_cap_b > truck_b.capacity:
                        continue
                    
                    # Create temporary routes for evaluation
                    route_a = truck_a.route.copy()
                    route_b = truck_b.route.copy()
                    
                    # Swap nodes
                    route_a[pos_a] = node_b
                    route_b[pos_b] = node_a
                    
                    # Calculate savings
                    orig_dist_a = calculate_route_metric(truck_a.route, matrix, mapper, cache)
                    orig_dist_b = calculate_route_metric(truck_b.route, matrix, mapper, cache)
                    new_dist_a = calculate_route_metric(route_a, matrix, mapper, cache)
                    new_dist_b = calculate_route_metric(route_b, matrix, mapper, cache)
                    
                    saving = orig_dist_a + orig_dist_b - new_dist_a - new_dist_b
                    
                    if saving > 0:
                        savings.append({
                            'saving': saving,
                            'truck_a': truck_a,
                            'truck_b': truck_b,
                            'node_a': node_a,
                            'node_b': node_b
                        })
    
    # Sort savings by value
    savings.sort(key=lambda x: x['saving'], reverse=True)
    
    if savings:
    # Apply best feasible swap
        swap = max(savings, key=lambda x: x['saving']) # type: ignore
        truck_a = swap['truck_a']
        truck_b = swap['truck_b']
        node_a = swap['node_a']
        node_b = swap['node_b']
        
        swap_nodes(node_a, node_b, truck_a, truck_b)
        
        
@perturbations_timer
def transfer_node_savings(trucks: List[Truck], mapper: Dict[str,int], matrix: np.ndarray, cache: RouteCache) -> None:
    """
    Attempts to transfer nodes between trucks to achieve route savings.
    This function iterates over pairs of used trucks and evaluates the potential savings
    from transferring nodes (and associated drone demands) from one truck to another.
    If a transfer results in a positive saving, it is recorded. The best feasible transfer
    is then applied.
    Args:
        trucks (List[Truck]): List of Truck objects representing the fleet.
        mapper (Dict[str, int]): A dictionary mapping node identifiers to their indices in the distance matrix.
        matrix (np.ndarray): A 2D numpy array representing the distance matrix.
        cache (RouteCache): A cache object to store and retrieve precomputed route metrics.
    Returns:
        None
    """
    used_trucks = [truck for truck in trucks if truck.is_used]
    if len(used_trucks) < 2:
        return

    savings = []
    
    for i, truck_a in enumerate(used_trucks[:-1]):
        for pos_a in range(1, len(truck_a.route) - 1):
            node_to_transfer = truck_a.route[pos_a]
            
            # Calculate associated drone demand for parking lots
            associated_drone_demand = 0
            associated_drones = []
            if node_to_transfer.node_type == 'Parking Lot':
                associated_drones = [d for d in truck_a.drones if d.route[0] == node_to_transfer]
                associated_drone_demand = sum(d.weight for d in associated_drones)
            
            total_transfer_demand = node_to_transfer._demand + associated_drone_demand
            
            for truck_b in used_trucks[i+1:]:
                # Skip if capacity constraint would be violated
                if truck_b.used_capacity + total_transfer_demand <= truck_b.capacity:
                    continue
                
                # Create temporary routes
                new_route_a = truck_a.route[:]
                new_route_b = truck_b.route[:]
                
                # Remove node from route A and add to route B
                new_route_a.pop(pos_a)
                new_route_b.insert(len(new_route_b) - 1, node_to_transfer)
                
                # Calculate savings
                orig_dist_a = calculate_route_metric(truck_a.route, matrix, mapper, cache)
                orig_dist_b = calculate_route_metric(truck_b.route, matrix, mapper, cache)
                new_dist_a = calculate_route_metric(new_route_a, matrix, mapper, cache)
                new_dist_b = calculate_route_metric(new_route_b, matrix, mapper, cache)
                
                saving = orig_dist_a + orig_dist_b - new_dist_a - new_dist_b
                
                if saving > 0:
                    savings.append({
                        'saving': saving,
                        'truck_from': truck_a,
                        'truck_to': truck_b,
                        'node': node_to_transfer
                    })
                    
    # Sort savings by value
    savings.sort(key=lambda x: x['saving'], reverse=True)
    
    # Apply best feasible transfer
    if savings:
        transfer = max(savings, key=lambda x: x['saving']) # type: ignore
        truck_from = transfer['truck_from']
        truck_to = transfer['truck_to']
        node = transfer['node']
            
        # Transfer the node
        transfer_node(node, truck_from, truck_to)
    

@perturbations_timer
def swap_truck_drone_savings(truck: Truck, mapper: Dict[str,int], truck_matrix: np.ndarray, drone_matrix: np.ndarray, truck_cache: RouteCache, drone_cache: RouteCache):
    """
    Attempts to swap nodes between a truck and its associated drones to achieve emission savings.
    This function evaluates the potential emission savings by swapping customer nodes between a truck's route and its associated drones' routes. If a feasible swap is found that results in emission savings, the swap is performed.
    Args:
        truck (Truck): The truck object containing its route and associated drones.
        mapper (Dict[str, int]): A dictionary mapping node IDs to their respective indices in the distance matrices.
        truck_matrix (np.ndarray): The distance matrix for the truck.
        drone_matrix (np.ndarray): The distance matrix for the drones.
        truck_cache (RouteCache): A cache object for storing and retrieving precomputed truck route metrics.
        drone_cache (RouteCache): A cache object for storing and retrieving precomputed drone route metrics.
    Returns:
        None
    """
    
    savings: List[Dict[str, Union[float, Node, Drone]]] = []
    
    truck_nodes = [node for node in truck.route if node.node_type == 'Customer']
    drone_nodes = [node for drone in truck.drones for node in drone.route if node.node_type == 'Customer']
    original_emissions = calculate_route_metric(truck.route, truck_matrix, mapper, truck_cache) + sum(calculate_route_metric(drone.route, drone_matrix, mapper, drone_cache) for drone in truck.drones)
    
    for t_node in truck_nodes:
        for d_node in drone_nodes:
            current_drone = next((drone for drone in truck.drones if drone.visit_node == d_node), None)
            ## Check Feaseability
            if current_drone and t_node._demand <= current_drone.capacity and 2 * drone_matrix[mapper[current_drone.route[0].id], mapper[t_node.id]] <= current_drone.max_distance:
                candidate_route = truck.route.copy()
                candidate_route[candidate_route.index(t_node)] = d_node
                candidate_drone_route = [current_drone.route[0], t_node, current_drone.route[0]]
                new_emissions = calculate_route_metric(candidate_route, truck_matrix, mapper, truck_cache) \
                                + sum(calculate_route_metric(drone.route, drone_matrix, mapper, drone_cache) for drone in truck.drones) \
                                - calculate_route_metric(current_drone.route, drone_matrix, mapper, drone_cache) \
                                + calculate_route_metric(candidate_drone_route, drone_matrix, mapper, drone_cache)
                    
                saving = original_emissions - new_emissions
                    
                if saving > 0:  # Keep original thresholds
                    savings.append({
                        'saving': saving,
                        'truck_node': t_node,
                        'drone_node': d_node,
                        'drone': current_drone
                    })
    if savings:
        # Select best saving and perform swap
        best_saving = max(savings, key=lambda x: x['saving']) # type: ignore
        truck_node = best_saving['truck_node']
        drone_node = best_saving['drone_node']
        drone = best_saving['drone']
        swap_nodes(truck_node, drone_node, truck, drone)
        

@perturbations_timer
def launch_drone_savings(truck: Truck, drones: List[Drone], mapper: Dict[str, int], truck_matrix: np.ndarray, drone_matrix: np.ndarray, truck_cache: RouteCache, drone_cache: RouteCache):
    """
    Attempts to launch drones to deliver to customer nodes in the truck's route, aiming to reduce emissions.
    This function evaluates potential savings in emissions by launching drones from parking lots to customer nodes.
    It calculates the emissions for the current truck route and compares it with the emissions for a modified route
    where a drone delivers to a customer node. If the modified route results in lower emissions, the function records
    the savings and selects the best option.
    Parameters:
        truck (Truck): The truck object containing the current route and other relevant information.
        drones (List[Drone]): A list of available drones that can be used for delivery.
        mapper (Dict[str, int]): A dictionary mapping node IDs to their respective indices in the distance matrices.
        truck_matrix (np.ndarray): A 2D numpy array representing the distance matrix for the truck.
        drone_matrix (np.ndarray): A 2D numpy array representing the distance matrix for the drones.
        truck_cache (RouteCache): A cache object for storing and retrieving precomputed truck route metrics.
        drone_cache (RouteCache): A cache object for storing and retrieving precomputed drone route metrics.
    Returns:
        None
    """
    customer_nodes = [node for node in truck.route if node.node_type == 'Customer']
    parking_lots = [node for node in truck.route if node.node_type == 'Parking Lot']
    available_drones = [drone for drone in drones if not drone.is_used]
    
    savings: List[Dict[str, Union[float, Node, Drone, Node]]] = []
    
    for node in customer_nodes:
        for parking in parking_lots:
            drone = next((d for d in available_drones if 2 * drone_matrix[mapper[parking.id], mapper[node.id]] <= d.max_distance and d.capacity >= node._demand), None)
            if drone:
                candidate_drone_route = [parking, node, parking]
                candidate_truck_route = truck.route.copy()
                if parking in candidate_truck_route:
                    candidate_truck_route.remove(node)
                if parking not in candidate_truck_route:
                    candidate_truck_route[candidate_truck_route.index(node)] = parking
                new_emissions = calculate_route_metric(candidate_truck_route, truck_matrix, mapper, truck_cache) \
                                + calculate_route_metric(candidate_drone_route, drone_matrix, mapper, drone_cache)
                original_emissions = calculate_route_metric(truck.route, truck_matrix, mapper, truck_cache) \
                                    + sum(calculate_route_metric(drone.route, drone_matrix, mapper, drone_cache) for drone in truck.drones)
                saving = original_emissions - new_emissions
                if saving > 0:
                    savings.append({
                        'saving': saving,
                        'truck_node': node,
                        'drone': drone,
                        'parking': parking
                    })
    if savings:
        best_saving = max(savings, key=lambda x: x['saving']) # type: ignore
        truck_node = best_saving['truck_node']
        drone = best_saving['drone']
        parking = best_saving['parking']
        launch_drone(drone, truck, parking, truck_node)


@perturbations_timer
def drone_optimization(drone: Drone, nodes: List[Node], mapper: Dict[str,int], truck_matrix: np.ndarray, drone_matrix: np.ndarray, truck_cache: RouteCache, drone_cache: RouteCache):
    parking_lots = [node for node in nodes if node.node_type == 'Parking Lot']
    
    