"""
This module contains functions for constructing initial solutions for the LRP-D.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

from models.nodes import Node
from models.solution import Solution
from models.vehicles import Drone, Truck
from utils.metrics import RouteCache, calculate_route_metric
from utils.timer import FunctionTimer
from dataclasses import dataclass

@dataclass
class Saving:
    i: int
    j: int
    value: float

construction_timer = FunctionTimer()

def nn_bigroute(base_nodes: List[Node], mapper: dict, distance_matrix: np.ndarray) -> List[Node]:
    """
    Constructs a route using the nearest neighbor algorithm.
    Args:
        base_nodes (List[Node]): A list of Node objects representing the nodes to be visited.
        mapper (dict): A dictionary mapping node IDs to their indices in the distance matrix.
        distance_matrix (np.ndarray): A 2D numpy array representing the distances between nodes.
    Returns:
        List[Node]: A list of Node objects representing the constructed route.
    """

    nodes = [node for node in base_nodes if node._node_type != 'Parking Lot']
    ncustomers = len([node for node in nodes if node._node_type == 'Customer'])
    
    # Initialize route
    route = [nodes[0]]
    visited = {nodes[0]}
    
    while len(visited) < ncustomers + 1:
        current = route[-1]
        current_index = mapper[current.id]
        
        next_node = min(
            (node for node in nodes if node not in visited),
            key=lambda x: distance_matrix[current_index, mapper[x.id]] # type: ignore
        ) # type: ignore
        route.append(next_node)
        visited.add(next_node)
    
    route.append(nodes[0])
    return route

def cw_bigroute(base_nodes: List[Node], mapper: dict, distance_matrix: np.ndarray) -> List[Node]:
    """
    Implements the Clarke-Wright Savings algorithm to construct a single route for a vehicle routing problem.
    Args:
        base_nodes (List[Node]): List of all nodes including depot, customers, and parking lots.
        mapper (dict): Dictionary mapping node IDs to their indices in the distance matrix.
        distance_matrix (np.ndarray): 2D array representing the distances between nodes.
    Returns:
        List[Node]: The final route starting and ending at the depot, visiting all customers.
    """
    # Filter out parking lots and get depot/customers
    nodes = [node for node in base_nodes if node._node_type != 'Parking Lot']
    depot = next(node for node in nodes if node._node_type == '0')
    customers = [node for node in nodes if node._node_type == 'Customer']
    
    # Calculate savings for all customer pairs
    savings = []
    depot_idx = mapper[depot.id]
    
    for i, node1 in enumerate(customers):
        i_idx = mapper[node1.id]
        for j, node2 in enumerate(customers[i+1:], i+1):
            j_idx = mapper[node2.id]
            
            # Savings formula: s_ij = c_i0 + c_0j - c_ij
            # where 0 is depot, i and j are customers
            saving = (
                distance_matrix[i_idx, depot_idx] +
                distance_matrix[depot_idx, j_idx] -
                distance_matrix[i_idx, j_idx]
            )
            
            savings.append(Saving(i_idx, j_idx, saving))
    
    # Sort savings in descending order
    savings.sort(key=lambda x: x.value, reverse=True)
    
    # Initialize routes: each customer is in a separate route
    routes: Dict[int, List[int]] = {
        mapper[customer.id]: [mapper[customer.id]]
        for customer in customers
    }
    
    # Track the start and end nodes of each route
    route_ends: Dict[int, Tuple[int, int]] = {
        mapper[customer.id]: (mapper[customer.id], mapper[customer.id])
        for customer in customers
    }
    
    # Merge routes based on savings
    for saving in savings:
        route1 = None
        route2 = None
        
        # Find which routes contain the nodes i and j
        for route_id, (start, end) in route_ends.items():
            if start == saving.i or end == saving.i:
                route1 = route_id
            if start == saving.j or end == saving.j:
                route2 = route_id
        
        # Skip if nodes are already in the same route
        if route1 == route2 or route1 is None or route2 is None:
            continue
            
        # Merge the routes
        route1_points = routes[route1]
        route2_points = routes[route2]
        
        # Determine how to connect the routes based on which ends match
        start1, end1 = route_ends[route1]
        start2, end2 = route_ends[route2]
        
        if end1 == saving.i and start2 == saving.j:
            new_route = route1_points + route2_points
            new_ends = (start1, end2)
        elif end1 == saving.i and end2 == saving.j:
            new_route = route1_points + route2_points[::-1]
            new_ends = (start1, start2)
        elif start1 == saving.i and start2 == saving.j:
            new_route = route1_points[::-1] + route2_points
            new_ends = (end1, end2)
        elif start1 == saving.i and end2 == saving.j:
            new_route = route1_points[::-1] + route2_points[::-1]
            new_ends = (end1, start2)
        else:
            continue
        
        # Update the routes and ends
        routes[route1] = new_route
        route_ends[route1] = new_ends
        
        # Remove the second route
        del routes[route2]
        del route_ends[route2]
    
    # Get the final route
    final_route_indices = next(iter(routes.values()))
    
    # Convert indices back to nodes and add depot at start/end
    final_route = []
    reverse_mapper = {v: k for k, v in mapper.items()}
    
    final_route.append(depot)
    for idx in final_route_indices:
        node_id = reverse_mapper[idx]
        node = next(node for node in nodes if node.id == node_id)
        final_route.append(node)
    final_route.append(depot)
    
    return final_route

def assign_drones_bigroute(base_nodes: List[Node], bigroute: List[Node], drones: List[Drone], mapper: Dict[str, int], truck_matrix: np.ndarray, drone_matrix: np.ndarray) -> None:
    """
    Assigns drones to deliver to customer nodes in a given bigroute, optimizing for emission savings.
    Parameters:
    base_nodes (List[Node]): List of base nodes, including parking lots.
    bigroute (List[Node]): The main route that the truck will follow.
    drones (List[Drone]): List of available drones.
    mapper (Dict[str, int]): A dictionary mapping node IDs to their indices in the distance matrices.
    truck_matrix (np.ndarray): Distance matrix for the truck.
    drone_matrix (np.ndarray): Distance matrix for the drones.
    Returns:
    None: The function modifies the drones' routes and the bigroute in place.
    """
    cache = RouteCache()
    parking_lots = {node for node in base_nodes if node.node_type == 'Parking Lot'}
    
    # Precompute route indices and initial emissions
    node_indices = {node: idx for idx, node in enumerate(bigroute)}
    initial_emissions = calculate_route_metric(bigroute, truck_matrix, mapper, cache)
    
    # Calculate all valid savings
    savings = []
    for i, current in enumerate(bigroute[1:-1]):
        if current.node_type != 'Customer':
            continue
            
        for p_lot in parking_lots:
            for j, target in enumerate(bigroute[i+1:-1], i+1):
                if target.node_type != 'Customer' or target._demand > drones[0].capacity:
                    continue
                    
                new_route = bigroute.copy()
                new_route[j] = p_lot
                
                new_emissions = calculate_route_metric(new_route, truck_matrix, mapper, cache)
                drone_emissions = 2 * drone_matrix[mapper[p_lot.id], mapper[target.id]]

                
                saving = initial_emissions - (new_emissions + drone_emissions)
                if saving > 0:
                    savings.append({'parking_lot' : p_lot, 'customer_node' : target, 'saving' : saving})
    
    # Sort savings once
    savings.sort(reverse=True, key = lambda x: x['saving'])
    
    # Assign drones efficiently
    if savings:
        used_nodes: Set[Node] = set()
        for drone in drones:
            best_saving = next((saving for saving in savings if saving['customer_node'] not in used_nodes and saving['customer_node']._demand <= drone.capacity and 2*drone_matrix[mapper[saving['parking_lot'].id], mapper[saving['customer_node'].id]] <= drone.max_distance), None) # type: ignore
            if best_saving:
                lot = best_saving['parking_lot']
                visit_node = best_saving['customer_node']
                drone.route = [lot, visit_node, lot]
                drone.visit_node = visit_node
                used_nodes.add(visit_node)
                if lot not in bigroute:
                    bigroute[bigroute.index(visit_node)] = lot
                else:
                    bigroute.remove(visit_node)
            else:
                break

def find_available_truck(trucks: List[Truck], required_capacity: float) -> Optional[Truck]:
    """
    Find an available truck that can accommodate the required capacity.

    Args:
        trucks (List[Truck]): A list of Truck objects to search through.
        required_capacity (float): The capacity needed for the truck.

    Returns:
        Optional[Truck]: The first Truck object that can accommodate the required capacity,
                         or None if no such truck is found.
    """
    return next(
        (truck for truck in trucks 
         if truck.used_capacity + required_capacity <= truck.capacity),
        None
    )

def cluster_bigroute(bigroute: List[Node], trucks: List[Truck], base_nodes: List[Node], drones: List[Drone]) -> None:
    """
    Distributes nodes from a big route among multiple trucks and assigns drones to trucks based on their routes.
    Args:
        bigroute (List[Node]): The complete route including the depot at the start and end.
        trucks (List[Truck]): List of available trucks for node assignments.
        base_nodes (List[Node]): List of base nodes, with the first node being the depot.
        drones (List[Drone]): List of drones to be assigned to trucks based on their routes.
    Returns:
        None: The function modifies the trucks' routes and drone assignments in place.
    The function performs the following steps:
    1. Removes the depot from the big route to focus on customer and parking lot nodes.
    2. Calculates the target number of customer nodes per truck.
    3. Initializes assignments for each truck.
    4. Iterates through the nodes in the route, assigning them to trucks based on capacity and type.
    5. Handles any unassigned nodes by assigning them to the truck with the most available capacity.
    6. Builds the final routes for each truck, including the depot at the start and end.
    """
    depot = base_nodes[0]
    route_wt_depot = bigroute[1:-1]  # Route without depot
    
    # Calculate target nodes per truck
    n_customers = sum(1 for node in route_wt_depot if node.node_type == 'Customer')
    target_per_truck = max(1, n_customers // len(trucks))
    
    # Initialize assignments
    assignments = defaultdict(list)
    current_truck_idx = 0
    
    for node in route_wt_depot:
        if node.isVisited:
            continue
            
        current_truck = trucks[current_truck_idx]
        
        if node.node_type == 'Parking Lot':
            relevant_drones = [d for d in drones if node in d.route]
            total_demand = (node._demand + 
                          sum(d.weight + d.visit_node._demand for d in relevant_drones))
            
            # Try current truck first, then any available truck
            assigned_truck = (
                current_truck if current_truck.used_capacity + total_demand <= current_truck.capacity
                else find_available_truck(trucks, total_demand)
            )
            
            if assigned_truck:
                assignments[assigned_truck].append(node)
                for drone in relevant_drones:
                    assigned_truck.drones.append(drone)
                node.isVisited = True
                
        else:  # Customer node
            assigned_truck = (
                current_truck if current_truck.used_capacity + node._demand <= current_truck.capacity
                else find_available_truck(trucks, node._demand)
            )
            
            if assigned_truck:
                assignments[assigned_truck].append(node)
                node.isVisited = True
                
                # Move to next truck if target reached
                if (assigned_truck == current_truck and 
                    len([n for n in assignments[current_truck] 
                         if n.node_type == 'Customer']) >= target_per_truck):
                    current_truck_idx = (current_truck_idx + 1) % len(trucks)
    
    # Handle any unassigned nodes
    unassigned = [node for node in route_wt_depot if not node.isVisited]
    for node in unassigned:
        # Find truck with most available capacity
        if best_truck := next(
            (t for t in sorted(
                trucks,
                key=lambda x: x.capacity - x.used_capacity,
                reverse=True
            ) if t.capacity > t.used_capacity),
            None
        ):
            assignments[best_truck].append(node)
            node.isVisited = True
    
    # Build final routes
    for truck in trucks:
        if nodes := assignments[truck]:
            # Sort based on original bigroute order
            sorted_nodes = sorted(nodes, key=lambda x: bigroute.index(x))
            truck.route = [depot] + sorted_nodes + [depot]
        else:
            truck.route = []


@construction_timer                                
def initialize_solution(instance_nodes: List[Node], mapper: Dict[str, int], truck_dm: np.ndarray, drone_dm: np.ndarray, instance_trucks: List[Truck], instance_drones: List[Drone]) -> Solution:
    """
    Initializes a solution for the given instance.
    Args:
        instance_nodes (List[Node]): List of nodes in the instance.
        mapper (Dict[str, int]): Mapping of node identifiers to indices.
        truck_dm (np.ndarray): Distance matrix for trucks.
        drone_dm (np.ndarray): Distance matrix for drones.
        instance_trucks (List[Truck]): List of trucks available in the instance.
        instance_drones (List[Drone]): List of drones available in the instance.
    Returns:
        Solution: The initial solution for the instance.
    """
    # Generate initial solution
    #bigroute = nn_bigroute(instance_nodes, mapper, truck_dm)
    bigroute = cw_bigroute(instance_nodes, mapper, truck_dm)
    
    assign_drones_bigroute(instance_nodes, bigroute, instance_drones, mapper, truck_dm, drone_dm)
    
    # Cluster bigroute into truck routes
    cluster_bigroute(bigroute, instance_trucks, instance_nodes, instance_drones)
    
    initial_sol = Solution(0,instance_nodes, instance_trucks, instance_drones)
    

    return initial_sol