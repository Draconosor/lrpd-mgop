from typing import List

from models.nodes import Nodes


class Vehicle:
    """Base vehicle class with common attributes."""
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        self._id = id
        self._capacity = capacity
        self._emissions = emissions
        self.route: List[Nodes] = []
        self.used_capacity: float = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def emissions(self) -> float:
        return self._emissions

    @property
    def pct_used_capacity(self) -> float:
        return round(self.used_capacity * 100 / self.capacity, 2)

    @property
    def is_used(self) -> bool:
        return self.used_capacity > 0

    def reset_vehicle(self):
        self.route = []
        self.used_capacity = 0

class Drone(Vehicle):
    """Specialized drone vehicle with additional attributes."""
    def __init__(self, id: str, capacity: float, emissions: float, max_distance: float, weight: float) -> None:
        super().__init__(id, capacity, emissions)
        self._max_distance = max_distance
        self._weight: float = weight
        self.assigned_to: str = ''
        self.visit_node: Nodes = None
    
    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Drone {self.id} ({self.pct_used_capacity}% capacity) {f'with route: {route_str} assigned to {self.assigned_to}' if not self.route == [] else ''}"

    def reset_vehicle(self):
        super().reset_vehicle()
        self.assigned_to = ''
        self.visit_node = None

    @property
    def max_distance(self) -> float:
        return self._max_distance

    @property
    def weight(self) -> float:
        return self._weight

class Truck(Vehicle):
    """Specialized truck vehicle that can carry drones."""
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        super().__init__(id, capacity, emissions)
        self.drones: List[Drone] = []
        
    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Truck {self.id}"
    
    def reset_vehicle(self):
        super().reset_vehicle()
        self.drones = []