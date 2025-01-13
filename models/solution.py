"""
This module contains the Solution class, which represents a complete solution to the routing problem.
"""
from typing import Dict, List

import numpy as np

from models.nodes import Node
from models.vehicles import Drone, Truck
from utils.metrics import RouteCache, calculate_route_metric


class Solution:
    """Represents a complete solution to the routing problem."""
    def __init__(self, id: int, nodes: List[Node], trucks: List[Truck], drones: List[Drone]) -> None:
        self.id = id
        self.nodes = nodes
        self.trucks = trucks
        self.drones = drones
        self.last_improvement: str = ''
        
    def __repr__(self):
        return f'Solution {self.id}'
    
    def emissions(self, mapper: Dict[str, int], truck_matrix: np.ndarray, drone_matrix: np.ndarray) -> float:
        truck_cache = RouteCache()
        drone_cache = RouteCache()
        
        truck_emissions = sum(calculate_route_metric(truck.route, truck_matrix, mapper, truck_cache) 
                            for truck in self.trucks)
        drone_emissions = sum(calculate_route_metric(drone.route, drone_matrix, mapper, drone_cache) 
                            for truck in self.trucks for drone in truck.drones)
        return truck_emissions + drone_emissions
    
    def makespan_per_truck(self, mapper: Dict[str, int], truck_time_matrix: np.ndarray, 
                          drone_time_matrix: np.ndarray) -> Dict[Truck, float]:
        drone_cache = RouteCache()
        truck_makespan: Dict[Truck, float] = {}
        
        for truck in self.trucks:
            if not truck.is_used:
                continue
                
            current_time = 0
            prev_node = None
            
            for node in truck.route:
                if prev_node is not None:
                    current_time += truck_time_matrix[mapper[prev_node.id], mapper[node.id]]
                    
                if node.node_type == 'Parking Lot':
                    max_drone_time = 0
                    for drone in truck.drones:
                        if drone.route[0] == node and drone.route[-1] == node:
                            drone_time = calculate_route_metric(
                                drone.route, drone_time_matrix, mapper, drone_cache)
                            max_drone_time = max(max_drone_time, drone_time)
                    
                    current_time += max_drone_time
                    
                prev_node = node
                
            truck_makespan[truck] = current_time
        
        return truck_makespan
    
    def integrity_check(self):
        customers = {node for node in self.nodes if node.node_type == 'Customer'}
        visit_nodes = [node for truck in self.trucks for node in truck.route] + [drone.visit_node for truck in self.trucks for drone in truck.drones]
        for c in customers:
            count = 0
            for node in visit_nodes:
                if c == node:
                    count += 1
            if count != 1:
                raise Exception(f'Solution did not pass integrity check due to node {c}')