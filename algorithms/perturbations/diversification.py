"""
Diversification perturbations for the LRP-D.
"""
import random
from typing import Dict, List

import numpy as np

from algorithms.perturbations.decorators import perturbations_timer
from models.nodes import Node
from models.vehicles import (Drone, Truck, launch_drone, swap_nodes,
                             transfer_node)
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

@perturbations_timer
def add_parking(truck: Truck, base_nodes: List[Node]):
    """
    Adds a parking lot to the truck's route if it is not already present.

    This function adds a parking lot to the truck's route if it is not already present. The parking lot is chosen
    randomly from the list of base nodes.

    Args:
        truck (Truck): The truck object to which the parking lot will be added.
        base_nodes (List[Node]): A list of nodes from which the parking lot will be chosen.

    Returns:
        None
    """
    while True:
        parking_lots = [node for node in base_nodes if node.node_type == 'Parking Lot']
        selected_parking = random.choice(parking_lots)
        if selected_parking not in truck.route:
            truck.route.insert(1, selected_parking)
            for drone in truck.drones:
                p = np.random.random()
                if p < 0.5:
                    drone.route[0] = selected_parking
                    drone.route[-1] = selected_parking
            break
    
        