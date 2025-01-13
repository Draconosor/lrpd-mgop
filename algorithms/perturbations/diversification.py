"""
Diversification perturbations for the LRP-D.
"""
import random
from typing import Dict, List, Set, Union

import numpy as np

from algorithms.perturbations.decorators import perturbations_timer
from models.nodes import Node
from models.vehicles import (Drone, Truck, swap_nodes, transfer_node)
from utils.metrics import RouteCache, calculate_route_metric

def random_truck_paired_select(trucks: List[Truck]) -> List[Truck]:
    """
    Selects a random permutation of trucks that are used and have a route length of at least 3.

    Args:
        trucks (List[Truck]): A list of Truck objects.

    Returns:
        List[Truck]: A list of Truck objects that are used and have a route length of at least 3, 
                     in a randomly permuted order.
    """
    used_trucks = [truck for truck in trucks if truck.is_used and len(truck.route) >= 3]
    indices = np.arange(len(used_trucks))
    permuted_indices = np.random.permutation(indices)
    return [used_trucks[i] for i in permuted_indices]

@perturbations_timer
def swap_interruta_random(truck_a: Truck, truck_b: Truck):
    """
    Attempts to swap a random node from the route of truck_a with a random node from the route of truck_b.
    The swap is only performed if the new capacities of both trucks do not exceed their respective limits.
    The function will try up to 50 times to find a valid swap.

    Args:
        truck_a (Truck): The first truck involved in the swap.
        truck_b (Truck): The second truck involved in the swap.

    Returns:
        None
    """
    failures = 0
    success = False
    while not success and failures <= 50:
        node_a: Node = random.choice(truck_a.route[1:-1])
        node_b: Node = random.choice(truck_b.route[1:-1])
        demand_a = node_a._demand
        demand_b = node_b._demand
        drones_to_transfer_a = []
        drones_to_transfer_b = []
        if node_a.node_type == 'Parking Lot':
            drones_to_transfer_a = [d for d in truck_a.drones if d.route[0] == node_a]
            demand_a += sum([d.weight for d in drones_to_transfer_a])
        if node_b.node_type == 'Parking Lot':
            drones_to_transfer_b = [d for d in truck_b.drones if d.route[0] == node_b]
            demand_b += sum([d.weight for d in drones_to_transfer_b])
        new_capacity_a = truck_a.used_capacity - demand_a + demand_b
        new_capacity_b = truck_b.used_capacity - demand_b + demand_a
        if new_capacity_a <= truck_a.capacity and new_capacity_b <= truck_b.capacity:
            swap_nodes(node_a, node_b, truck_a, truck_b)                                              
            success = True
        else:
            failures += 1

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
def shuffle_route(truck: Truck):
    """
    Shuffles the route of the given truck, excluding the depot nodes.

    This function takes a truck object and shuffles its route, excluding the 
    first and last nodes (assumed to be the depot). The shuffled route is then 
    assigned back to the truck. If the first or last node of the route is not 
    the depot after shuffling, an exception is raised.

    Args:
        truck (Truck): The truck object whose route is to be shuffled. The 
        truck object must have a 'route' attribute which is a list of nodes, 
        and each node must have an 'id' attribute.
    """
    if len(truck.route) > 3:
        node_0 = truck.route[0]
        route_wt_depot = truck.route[1:-1]
        random.shuffle(route_wt_depot)
        route = [node_0] + route_wt_depot + [node_0]
        truck.route = route
        if (truck.route[0].id != '0' or truck.route[-1].id != '0'):
            raise Exception
        
@perturbations_timer
def transfer_node_random(truck_from: Truck, truck_to: Truck):
    """
    Attempts to transfer a random node from one truck's route to another truck's route.
    This function will try up to 50 times to transfer a node from `truck_from` to `truck_to`.
    The node to be transferred is chosen randomly from the route of `truck_from`, excluding the first and last nodes.
    If the node to be transferred is a 'Parking Lot', the associated drone demand is also considered in the transfer.
    Args:
        truck_from (Truck): The truck from which a node will be transferred.
        truck_to (Truck): The truck to which the node will be transferred.
    Returns:
        None
    """
    failure = 0
    success = False
    if len(truck_from.route) > 3:
        while not success and failure < 50:
            node_to_transfer = random.choice(truck_from.route[1:-1])
            
            # Calculate associated drone demand for parking lots
            associated_drone_demand = 0
            associated_drones = []
            if node_to_transfer.node_type == 'Parking Lot':
                associated_drones = [d for d in truck_from.drones if d.route[0] == node_to_transfer]
                associated_drone_demand = sum(d.weight for d in associated_drones)
            
            total_transfer_demand = node_to_transfer._demand + associated_drone_demand
            
            ## Check Feaseability
            
            if truck_to.used_capacity + total_transfer_demand <= truck_to.capacity:
                transfer_node(node_to_transfer, truck_from,truck_to)
                success = True
            failure += 1
            
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
def group_parkings(truck: Truck, mapper: Dict[str, int], truck_matrix: np.ndarray, drone_matrix: np.ndarray) -> None:
    """
    Groups drones to the most optimal parking lot in the truck's route to minimize emissions.
    This function evaluates each parking lot in the truck's route to determine which one can serve the most drones
    while minimizing the overall emissions. It updates the routes of the drones to start and end at the selected
    parking lot if it results in lower emissions.
    Args:
        truck (Truck): The truck object containing its route and associated drones.
        mapper (Dict[str, int]): A dictionary mapping node IDs to their respective indices in the distance matrices.
        truck_matrix (np.ndarray): A 2D numpy array representing the distance matrix for the truck.
        drone_matrix (np.ndarray): A 2D numpy array representing the distance matrix for the drones.
    Returns:
        None: The function updates the routes of the drones in place if a more optimal parking lot is found.
    """
    # Get all parking lots from truck route
    parking_lots = [node for node in truck.route if node.node_type == 'Parking Lot']
    
    if not parking_lots:
        return  # No parking lots to process
        
    current_emissions = sum(calculate_route_metric(d.route, drone_matrix, mapper, RouteCache()) for d in truck.drones) + calculate_route_metric(truck.route, truck_matrix, mapper, RouteCache())
    
    # Try each parking lot to find the one that can serve most drones
    best_parking = None
    best_feasible_drones = []
    
    for candidate_parking in parking_lots:
        feasible_drones = [drone for drone in truck.drones if 2 * drone_matrix[mapper[candidate_parking.id], mapper[drone.visit_node.id]] <= drone.max_distance]
        unfeasible_drone_routes = [drone.route for drone in truck.drones if 2 * drone_matrix[mapper[candidate_parking.id], mapper[drone.visit_node.id]] > drone.max_distance]
        feasible_visit_nodes = [drone.visit_node for drone in feasible_drones]
        test_routes = [[candidate_parking, node, candidate_parking] for node in feasible_visit_nodes] + unfeasible_drone_routes
        new_drone_emissions = sum(calculate_route_metric(r, drone_matrix, mapper, RouteCache()) for r in test_routes)
        test_truck_route = truck.route.copy()
        for node in test_truck_route:
            if node.node_type == 'Parking Lot' and node not in [route[0] for route in test_routes]:
                test_truck_route.remove(node)
        new_truck_emissions = calculate_route_metric(test_truck_route, truck_matrix, mapper, RouteCache())
        new_emissions = new_drone_emissions + new_truck_emissions
        if new_emissions < current_emissions:
            best_parking = candidate_parking
            current_emissions = new_emissions
            best_feasible_drones = feasible_drones
    
    if best_parking is None:
        return  # No feasible grouping found
    
    for drone in truck.drones:
        if drone in best_feasible_drones:
            if drone.route[0] != best_parking:
                drone.route = [best_parking, drone.visit_node, best_parking]
            
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
def fuse_trucks(truck_a: Truck, truck_b: Truck):
    """
    Attempts to fuse the routes of two trucks by transferring nodes from one truck to another if feasible.

    This function checks if the combined used capacity of both trucks does not exceed the capacity of either truck.
    If feasible, it transfers nodes from the less occupied truck (donor) to the more occupied truck (receiver).

    Args:
        truck_a (Truck): The first truck involved in the fusion.
        truck_b (Truck): The second truck involved in the fusion.

    Returns:
        None
    """
    # Check feaseability
    if truck_a.used_capacity + truck_b.used_capacity <= truck_a.capacity or truck_a.used_capacity + truck_b.used_capacity <= truck_b.capacity:     
        # Get less occuped_truck
        receiver: Truck = min([truck_a, truck_b], key = lambda x: x.used_capacity)
        donor: Truck = [truck for truck in [truck_a, truck_b] if truck != receiver][0]
        # Pass nodes
        for node in donor.route[1:-1]:
            transfer_node(node, donor, receiver)
        