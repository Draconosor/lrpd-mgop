from typing import Dict, List

import numpy as np
import pandas as pd

from models.nodes import Nodes
from models.vehicles import Drone, Truck
from utils.metrics import RouteCache, calculate_route_metric


class Saving:
    def __init__(self, parking_lot: Nodes, to: Nodes, saving: float, prev_node: Nodes = None, truck_a: Truck = None, truck_b: Truck = None) -> None:
        self.parking_lot = parking_lot
        self.to = to
        self.prev_node = prev_node
        self._saving = saving
        self.truck_a = truck_a
        self.truck_b = truck_b
        self.used = False
        
    @property
    def saving(self):
        return self._saving
    
    @property
    def strict_nodes(self):
        return [self.parking_lot, self.to]
        
    def __repr__(self) -> str:
        return f'Saving from {self.prev_node} to parking {self.parking_lot.id} using drone to {self.to.id} of value {self.saving}'
    
    def __eq__(self, other):
        if isinstance(other, Saving):
            return self.saving == other.saving
        return False
    
    def __lt__(self, other):
        if not isinstance(other, Saving):
            return NotImplemented
        return self.saving < other.saving

    def __gt__(self, other):
        if not isinstance(other, Saving):
            return NotImplemented
        return self.saving > other.saving


class Solution:
    """Represents a complete solution to the routing problem."""
    def __init__(self, id: int, nodes: List[Nodes], trucks: List[Truck], drones: List[Drone]) -> None:
        self.id = id
        self.nodes = nodes
        self.trucks = trucks
        self.drones = drones
        
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