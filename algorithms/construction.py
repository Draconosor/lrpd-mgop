from typing import List, Dict
from models.nodes import Nodes
from models.vehicles import Truck, Drone
from models.solution import Solution, Saving
import pandas as pd
from utils.metrics import calculate_route_metric, RouteCache
from algorithms.improvement import twoopt_until_local_optimum
from collections import defaultdict
import numpy as np

def nn_bigroute(base_nodes: List[Nodes], mapper: dict, distance_matrix: np.ndarray) -> List[Nodes]:
    """Optimized Nearest-Neighbor algorithm for Big Route generation"""
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

def assign_drones_bigroute(base_nodes: List[Nodes], bigroute: List[Nodes], 
                          drones: List[Drone], mapper: Dict[str, int], truck_matrix: np.ndarray, 
                          drone_matrix: np.ndarray) -> None:
    """Optimized drone assignment to bigroute"""
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
                    savings.append(Saving(p_lot, target, saving))
    
    # Sort savings once
    savings.sort(reverse=True)
    
    # Assign drones efficiently
    used_nodes = set()
    for drone in drones:
        for saving in savings:
            if (not saving.used and saving.to not in used_nodes and 
                saving.to._demand <= drone.capacity and 
                2 * drone_matrix[mapper[saving.parking_lot.id], mapper[saving.to.id]] <= drone.max_distance):
                
                fly_to = saving.to
                p_lot = saving.parking_lot
                
                # Update drone
                drone.visit_node = fly_to
                drone.route = [p_lot, fly_to, p_lot]
                drone.used_capacity = fly_to._demand
    
                if p_lot._demand > 0:
                    bigroute.pop(bigroute.index(fly_to))
                else:
                    bigroute[bigroute.index(fly_to)] = p_lot
                
                # Update parking lot and route
                p_lot._demand += fly_to._demand
                saving.used = True
                used_nodes.add(fly_to)
                break
            
def cluster_bigroute(bigroute: List[Nodes], trucks: List[Truck], base_nodes: List[Nodes], drones: List[Drone]) -> None:
    """Optimized clustering of bigroute into truck routes"""
    depot = base_nodes[0]
    route_wt_depot = bigroute[1:-1]
    
    # Track assignments
    assignments = defaultdict(list)
    drone_assignments = defaultdict(list)
    
    for node in route_wt_depot:
        assigned = False
        for truck in trucks:
            if not node.isVisited and truck.used_capacity + node._demand <= truck.capacity:
                if node.node_type == 'Parking Lot':
                    relevant_drones = [d for d in drones if node in d.route]
                    if all(truck.used_capacity + d.weight + d.visit_node._demand <= truck.capacity 
                          for d in relevant_drones):
                        for drone in relevant_drones:
                            drone.assigned_to = truck.id
                            drone_assignments[truck].append(drone)
                            truck.used_capacity += drone.weight + drone.visit_node._demand
                
                assignments[truck].append(node)
                truck.used_capacity += node._demand
                node.isVisited = True
                assigned = True
                break
        
        if not assigned:
            # Handle unassigned nodes
            for truck in trucks:
                if not truck.is_used:
                    assignments[truck].append(node)
                    truck.used_capacity += node._demand
                    node.isVisited = True
                    break
    
    # Build final routes
    for truck in trucks:
        if assignments[truck]:
            truck.route = [depot] + assignments[truck] + [depot]
            truck.drones = drone_assignments[truck]

def initialize_solution(instance_nodes: List[Nodes], mapper: Dict[str, int], 
                        truck_dm: np.ndarray, drone_dm: np.ndarray, 
                        instance_trucks: List[Truck], instance_drones: List[Drone]) -> Solution:
    
    # Create route cache for the entire solution process
    truck_cache = RouteCache()
    
    # Generate initial solution
    bigroute = nn_bigroute(instance_nodes, mapper, truck_dm)
    
    # Assign drones to bigroute
    assign_drones_bigroute(instance_nodes, bigroute, instance_drones, mapper, truck_dm, drone_dm)
    
    # Cluster bigroute into truck routes
    cluster_bigroute(bigroute, instance_trucks, instance_nodes, instance_drones)
    
    # Optimize all truck routes
    for truck in instance_trucks:
        if truck.route:
            truck.route = twoopt_until_local_optimum(truck.route, truck_dm, mapper, truck_cache)
    
    initial_sol = Solution(0,instance_nodes, instance_trucks, instance_drones)
    
    return initial_sol