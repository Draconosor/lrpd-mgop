""" This module contains the Vehicle, Drone, and Truck classes for the routing problem. """
from typing import List, Union

from models.nodes import Node


class Vehicle:
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        """
        Initialize a Vehicle instance.

        Args:
            id (str): The unique identifier for the vehicle.
            capacity (float): The capacity of the vehicle.
            emissions (float): The emissions rate of the vehicle.

        Attributes:
            _id (str): The unique identifier for the vehicle.
            _capacity (float): The capacity of the vehicle.
            _emissions (float): The emissions rate of the vehicle.
            route (List[Node]): The route assigned to the vehicle.
        """
        self._id = id
        self._capacity = capacity
        self._emissions = emissions
        self.route: List[Node] = []

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
    def used_capacity(self):
        """
        Calculate the used capacity of the vehicle.

        This property sums up the demand of all nodes in the vehicle's route.

        Returns:
            int: The total used capacity of the vehicle.
        """
        return sum(node._demand for node in self.route) # type: ignore
    
    @property
    def is_used(self) -> bool:
        return self.used_capacity > 0

    def reset_vehicle(self):
        """
        Resets the vehicle's state by clearing its route.

        This method clears the current route of the vehicle, effectively resetting
        its state to an initial condition where no route is assigned.
        """
        self.route = []

class Drone(Vehicle):
    """Specialized drone vehicle with additional attributes."""
    def __init__(self, id: str, capacity: float, emissions: float, max_distance: float, weight: float) -> None:
        """
        Initialize a Drone instance.

        Args:
            id (str): The unique identifier for the Drone.
            capacity (float): The capacity of the Drone.
            emissions (float): The emissions produced by the Drone.
            max_distance (float): The maximum distance the Drone can travel.
            weight (float): The weight of the Drone.

        Attributes:
            _max_distance (float): The maximum distance the Drone can travel.
            _weight (float): The weight of the Drone.
            assigned_to (str): The identifier of the entity to which the Drone is assigned.
            visit_node (Node): The node that the Drone is assigned to visit.
        """
        super().__init__(id, capacity, emissions)
        self._max_distance = max_distance
        self._weight: float = weight
        self.assigned_to: str = ''
        self.visit_node: Node = None # type: ignore
    
    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Drone {self.id} ({self.pct_used_capacity}% capacity) {f'with route: {route_str} assigned to {self.assigned_to}' if not self.route == [] else ''}"
    
    def reset_vehicle(self):
        """
        Resets the Drone to its initial state.

        This method overrides the parent class's reset_vehicle method to 
        reset additional attributes specific to this Drone.

        Attributes reset:
            assigned_to (str): The identifier of the entity to which the Drone is assigned.
            visit_node (Any): The node that the Drone is set to visit. Set to None.
        """
        super().reset_vehicle()
        self.assigned_to = ''
        self.visit_node = None # type: ignore

    @property
    def max_distance(self) -> float:
        return self._max_distance

    @property
    def weight(self) -> float:
        return self._weight

class Truck(Vehicle):
    """Specialized truck vehicle that can carry drones."""
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        """
        Initialize a Truck instance.

        Args:
            id (str): The unique identifier for the Truck.
            capacity (float): The capacity of the Truck.
            emissions (float): The emissions level of the Truck.

        Attributes:
            drones (List[Drone]): A list to store associated drones.
        """
        super().__init__(id, capacity, emissions)
        self.drones: List[Drone] = []
        
    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Truck {self.id} with route:{route_str} and Drones {self.drones}"
    
    @property
    def used_capacity(self):
        return super().used_capacity + sum(sum((drone.weight, drone.used_capacity)) for drone in self.drones)
    
    def reset_vehicle(self):
        super().reset_vehicle()
        self.drones.clear()
    
    def drop_unused_parkings(self):
        """
        Removes unused parking lots from the route.

        This method iterates through the route and removes any parking lot nodes that are not used by any drone.
        A parking lot is considered used if it is the starting point of any drone's route.

        Returns:
            None
        """
        used_parkings = {lot for drone in self.drones if drone.is_used for lot in drone.route if lot == drone.route[0]}
        for node in self.route:
            if node.node_type == 'Parking Lot' and node not in used_parkings:
                self.route.remove(node)
    
    
def integrity_check(truck: Truck):
    """
    Performs an integrity check on the given truck's route and its drones' routes.
    This function ensures that each drone's starting point (parking lot) is part of the truck's route.
    It extracts the parking lots from the truck's route and the starting points of each drone's route,
    then verifies that there is a 1:1 relationship between them.
    Args:
        truck (Truck): The truck object containing its route and the drones it carries.
    Raises:
        Exception: If the starting points of the drones are not a subset of the truck's parking lots.
    """
    # Extract parking lots from the truck's route
    lots = {node for node in truck.route if node.node_type == 'Parking Lot'}

    # Extract the starting point (lot) for each drone
    drone_lots = {drone.route[0] for drone in truck.drones}

    # Ensure a 1:1 relationship between parking lots and drone lots
    if not drone_lots.issubset(lots):
        raise Exception(
            f'Integrity Check Failed for truck {truck}. '
            f'Parking lots in route: {lots}, Drone starting lots: {drone_lots}'
    )

        
def assign_multiple_drones(drones: List[Drone], truck: Truck):
    """
    Assigns multiple drones to a truck.

    This function takes a list of drones and a truck, and assigns all the drones
    to the given truck. It extends the truck's drone list with the provided drones
    and ensures that each drone's `assigned_to` attribute is set to the truck's ID.

    Args:
        drones (List[Drone]): A list of Drone objects to be assigned to the truck.
        truck (Truck): The Truck object to which the drones will be assigned.

    Returns:
        None
    """
    truck.drones.extend(drones)
    for d in truck.drones:
        if d.assigned_to != truck.id:
            d.assigned_to = truck.id
            
def remove_node(node: Node, vehicle: Union[Truck, Drone]):
    """
    Removes a node from the vehicle's route and resets the vehicle if the route has fewer than 3 nodes.

    Args:
        node (Node): The node to be removed from the vehicle's route.
        vehicle (Union[Truck, Drone]): The vehicle from which the node will be removed. It can be either a Truck or a Drone.

    Returns:
        None
    """
    vehicle.route.remove(node)
    if len(vehicle.route) < 3:
        vehicle.reset_vehicle()
            
def add_node(node: Node, vehicle: Union[Truck, Drone]):
    """
    Add a node to a vehicle while respecting capacity constraints and vehicle rules.
    
    Args:
        node: Node to be added
        vehicle: Vehicle (Truck or Drone) to add node to
    """
        # Handle drone reassignment for trucks
    
    # Only collect drones if the vehicle is a truck and node is a parking lot
    if node._node_type == 'Parking Lot' and isinstance(vehicle, Drone):
        raise Exception ('Trying to reallocate Parking to Drone')
    if node._demand + vehicle.used_capacity > vehicle.capacity:
        raise Exception(f'Trying to add node to vehicle: {vehicle} but capacity constraint is violated')
    if isinstance(vehicle, Truck):
        # For empty truck routes, initialize with depot
        vehicle.route.insert(1, node)
    else:  # Drone   
        # Add node to drone's route
        vehicle.route.insert(1, node)
        # Update visit node if not set
        vehicle.visit_node = node
    
def swap_nodes(node_a: Node, node_b: Node, vehicle_a: Union[Truck, Drone], vehicle_b: Union[Truck, Drone]):
    """
    Swap nodes between any combination of vehicles (trucks or drones).
    
    Args:
        node_a: Node from first vehicle to be swapped
        node_b: Node from second vehicle to be swapped
        vehicle_a: First vehicle (can be either Truck or Drone)
        vehicle_b: Second vehicle (can be either Truck or Drone)
    """

    # Handle drone reassignment for trucks
    transfer_drones_a = []
    transfer_drones_b = []
    
    # Only collect drones if the vehicle is a truck and node is a parking lot
    if isinstance(vehicle_a, Truck) and node_a.node_type == 'Parking Lot':
        transfer_drones_a.extend([drone for drone in vehicle_a.drones if drone.route[0] == node_a])
    if isinstance(vehicle_b, Truck) and node_b.node_type == 'Parking Lot':
        transfer_drones_b.extend([drone for drone in vehicle_b.drones if drone.route[0] == node_b])
    
    drone_weight_a = sum(sum((drone.weight, drone.used_capacity)) for drone in transfer_drones_a)
    drone_weight_b = sum(sum((drone.weight, drone.used_capacity)) for drone in transfer_drones_b)
    
    if (vehicle_a.used_capacity - node_a._demand - drone_weight_a + drone_weight_b + node_b._demand > vehicle_a.capacity) or (vehicle_b.used_capacity - node_b._demand - drone_weight_b + drone_weight_a + node_a._demand > vehicle_b.capacity):
        exp_a = (vehicle_a.used_capacity - node_a._demand + node_b._demand > vehicle_a.capacity)
        exp_b = (vehicle_b.used_capacity - node_b._demand + node_a._demand > vehicle_b.capacity)
        return_string = ''
        if exp_a:
            return_string += f' {vehicle_a} {vehicle_a.used_capacity} - {node_a} {node_a._demand} - DWA{drone_weight_a} + DWB {drone_weight_b} + {node_b} {node_b._demand} > {vehicle_a} {vehicle_a.capacity}'
        if exp_b:
            return_string += f' {vehicle_b} {vehicle_b.used_capacity} - {node_b} {node_b._demand} - DWA{drone_weight_b} + DWB {drone_weight_a} + {node_a} {node_a._demand} > {vehicle_b} {vehicle_b.capacity}'
        raise Exception(f'Violated swap constraints {return_string}')

    if len(vehicle_a.route) < 3 or len(vehicle_b.route) < 3:
        raise Exception(f"""Trying to swap with empty route
                        vehicle a {vehicle_a.id}: {vehicle_a.route}
                        vehicle b {vehicle_b.id}: {vehicle_b.route}""")



    # Perform swap based on vehicle types
    if isinstance(vehicle_a, Truck):
        pos_node_a = vehicle_a.route.index(node_a)
        vehicle_a.route[pos_node_a] = node_b
        vehicle_a.drones = [drone for drone in vehicle_a.drones if drone not in transfer_drones_a]
        assign_multiple_drones(transfer_drones_b, vehicle_a)
    else:  # Drone
        vehicle_a.route[1] = node_b
        vehicle_a.visit_node = node_b

    if isinstance(vehicle_b, Truck):
        pos_node_b = vehicle_b.route.index(node_b)
        vehicle_b.route[pos_node_b] = node_a
        vehicle_b.drones = [drone for drone in vehicle_b.drones if drone not in transfer_drones_b]
        assign_multiple_drones(transfer_drones_a, vehicle_b)
    else:  # Drone
        vehicle_b.route[1] = node_a
        vehicle_b.visit_node = node_a
    
    if isinstance(vehicle_a, Truck):
        integrity_check(vehicle_a)
    if isinstance(vehicle_b, Truck):
        integrity_check(vehicle_b)    

def transfer_node(node: Node, vehicle_from: Union[Truck, Drone], vehicle_to: Truck):
    """
    Transfer a node from one vehicle to another (either truck or drone).
    
    Args:
        node: Node to be transferred
        vehicle_from: Source vehicle (can be either Truck or Drone)
        vehicle_to: Target vehicle (can be either Truck or Drone)
    """
    
    if isinstance(vehicle_to, Drone):
        raise Exception('Trying to transfer to Drone!!')
    # Validate target vehicle has a route if it's a truck
    if len(vehicle_to.route) < 3:
        raise Exception(f"""Trying to transfer to empty truck
                        vehicle from: {vehicle_from.route}
                        vehicle to: {vehicle_to.route}""")

    transfer_drones = []
    if isinstance(vehicle_from, Truck) and node.node_type == 'Parking Lot':
        transfer_drones.extend([drone for drone in vehicle_from.drones if drone.route[0] == node])
    
    # Remove node from source vehicle
    remove_node(node, vehicle_from)
    
    if isinstance(vehicle_from, Truck):
        vehicle_from.drones = [drone for drone in vehicle_from.drones if drone not in transfer_drones]

    # Add node to target vehicle
    add_node(node, vehicle_to)
    assign_multiple_drones(transfer_drones, vehicle_to)
    
    if isinstance(vehicle_from, Truck):
        integrity_check(vehicle_from)
    if isinstance(vehicle_to, Truck):
        integrity_check(vehicle_to)    
    

def launch_drone(drone: Drone, truck: Truck, parking_lot: Node, visit_node: Node):
    """
    Launches a drone from a truck, setting its route and updating the truck's route.

    Args:
        drone (Drone): The drone to be launched.
        truck (Truck): The truck from which the drone is launched.
        parking_lot (Node): The node representing the parking lot where the drone starts and ends its route.
        visit_node (Node): The node that the drone will visit.

    Returns:
        None
    """
    drone.route = [parking_lot, visit_node, parking_lot]
    drone.visit_node = visit_node
    truck.drones.append(drone)
    if parking_lot not in truck.route:
        node_pos = truck.route.index(visit_node)
        truck.route[node_pos] = parking_lot
    else:
        remove_node(visit_node, truck)