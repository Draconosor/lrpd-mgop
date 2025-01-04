
import random
from typing import Dict, List

import numpy as np

from algorithms.improvement import twoopt_until_local_optimum
from models.nodes import Nodes
from models.solution import Solution
from models.vehicles import Truck
from utils.metrics import RouteCache, calculate_route_metric
from utils.timer import FunctionTimer

perturbations_timer = FunctionTimer()


def random_truck_paired_select(trucks: List[Truck]) -> List[Truck]:
    used_trucks = [truck for truck in trucks if truck.is_used and len(truck.route) >= 3]
    indices = np.arange(len(used_trucks))
    permuted_indices = np.random.permutation(indices)
    return [used_trucks[i] for i in permuted_indices]

@perturbations_timer
def swap_interruta_random(truck_a: Truck, truck_b: Truck, mapper: Dict[str, int], matrix: np.ndarray, cache: RouteCache):
    failures = 0
    success = False
    while not success and failures <= 50:
        node_a: Nodes = random.choice(truck_a.route[1:-1])
        node_b: Nodes = random.choice(truck_b.route[1:-1])
        if truck_a.used_capacity - node_a._demand + node_b._demand <= truck_a.capacity and truck_b.used_capacity - node_b._demand + node_a._demand <= truck_b.capacity:
            truck_a.route[truck_a.route.index(node_a)] = node_b
            truck_a.used_capacity = truck_a.used_capacity - node_a._demand + node_b._demand
            truck_b.route[truck_b.route.index(node_b)] = node_a
            truck_b.used_capacity = truck_b.used_capacity - node_b._demand + node_a._demand
            truck_a.route = twoopt_until_local_optimum(truck_a.route, matrix, mapper, cache)
            truck_b.route = twoopt_until_local_optimum(truck_b.route, matrix, mapper, cache)
            success = True
        else:
            failures += 1

@perturbations_timer
def open_truck(trucks: List[Truck], mapper: Dict[str, int], truck_dm: np.ndarray, truck_cache: RouteCache):
    for i, truck in enumerate(trucks):
        if not truck.is_used and i > 0 and len(trucks[i-1].route) > 3:
            node_0 = trucks[i-1].route[0]
            prev_truck_route = trucks[i-1].route[1:-1]
            # Find the middle index
            middle_index = len(prev_truck_route) // 2

            # Split the list into two halves
            first_half = prev_truck_route[:middle_index]
            second_half = prev_truck_route[middle_index:]
            
            # Assign Routes
            ## Previous Truck
            trucks[i-1].route = first_half
            trucks[i-1].route.append(node_0)
            trucks[i-1].route.insert(0, node_0)
            
            for node in second_half:
                trucks[i-1].used_capacity -= node._demand
            
            ## New Truck
            for drone in trucks[i-1].drones:
                if drone.is_used:
                    if drone.route[0] in second_half:
                        drone.assigned_to = truck.id
                        truck.drones.append(drone)
                        truck.used_capacity += drone.weight
                        trucks[i-1].used_capacity -= drone.weight
                        
            trucks[i-1].drones = [d for d in trucks[i-1].drones if d.assigned_to == trucks[i-1].id]
            twoopt_until_local_optimum(trucks[i-1].route, truck_dm, mapper, truck_cache)
            
            for node in second_half:
                truck.used_capacity += node._demand
                
            truck.route = second_half
            truck.route.append(node_0)
            truck.route.insert(0, node_0)
            twoopt_until_local_optimum(truck.route, truck_dm, mapper, truck_cache)
            
            break

@perturbations_timer
def close_parking_lot_random(trucks: List[Truck], mapper: Dict[str, int], truck_matrix: np.ndarray, drone_matrix: np.ndarray, truck_cache: RouteCache):
    truck: Truck = random.choice([t for t in trucks if t.is_used])
    lots = [p for p in truck.route if p.node_type == 'Parking Lot']
    if lots:
        to_close: Nodes = random.choice(lots)
        remaining = lots[:]
        remaining.remove(to_close)
        if remaining:
            for drone in truck.drones:
                if drone.route[0] == to_close:
                    ## Look for best node available
                    best_cost = 10e6
                    best_route = []
                    for p_lot in remaining:
                        route = [p_lot, drone.visit_node, p_lot]
                        cost = calculate_route_metric(route, drone_matrix, mapper, RouteCache())
                        if cost <= drone.max_distance and cost < best_cost:
                            best_cost = cost
                            best_route = route
                    if best_route:
                        drone.route = best_route
                    else:
                        truck.route.insert(-2, drone.visit_node)
                        drone.reset_vehicle()
    twoopt_until_local_optimum(truck.route, truck_matrix, mapper, truck_cache)

@perturbations_timer
def shuffle_route(truck: Truck, mapper:Dict[str,int], truck_dm: np.ndarray, truck_cache: RouteCache):
    if len(truck.route) > 3:
        node_0 = truck.route[0]
        route_wt_depot = truck.route[1:-1]
        random.shuffle(route_wt_depot)
        route = [node_0] + route_wt_depot + [node_0]
        truck.route = route
        twoopt_until_local_optimum(truck.route, truck_dm, mapper, truck_cache)
        
@perturbations_timer
def transfer_node_random(truck_a: Truck, truck_b: Truck, mapper:Dict[str, int], truck_dm: np.ndarray, truck_cache: RouteCache):
    if len(truck_a.route) > 3:
        while True:
            selector = random.randint(1, len(truck_a.route)-1)
            if truck_a.route[selector].node_type == 'Customer':
                selected_node = truck_a.route.pop(selector)
                truck_b.route.insert(-2, selected_node)
                twoopt_until_local_optimum(truck_a.route, truck_dm, mapper, truck_cache)
                break
            
@perturbations_timer
def group_parkings(truck: Truck, mapper: Dict[str,int], drone_matrix: np.ndarray, truck_dm: np.ndarray, truck_cache: RouteCache):
    lots = [p for p in truck.route if p.node_type == 'Parking Lot']
    if lots == []:
        return None
    selected_parking: Nodes = random.choice(lots)
    used_parkings = []
    for drone in truck.drones:
        if drone.route[0] != selected_parking:
            # Check feaseability
            cost = 2 * drone_matrix[mapper[selected_parking.id], mapper[drone.visit_node.id]]
            if cost <= drone.max_distance:
                drone.route[0]._demand -= drone.used_capacity
                drone.route = [selected_parking, drone.visit_node, selected_parking]
                selected_parking._demand =+ drone.used_capacity
            else:
                pass
            if drone.route[0] not in used_parkings:
                used_parkings.append(drone.route[0])
    for node in truck.route:
        if node.node_type == 'Parking Lot' and node not in used_parkings:
            truck.route.remove(node)
    twoopt_until_local_optimum(truck.route, truck_dm, mapper, truck_cache)
    
@perturbations_timer
def optimize_drone_assignments(solution: Solution, mapper:Dict[str,int], drone_matrix: np.ndarray, 
                            truck_matrix: np.ndarray, truck_cache: RouteCache, drone_cache: RouteCache) -> None:
    """
    Optimizes drone assignments to minimize emissions by:
    1. Identifying inefficient drone routes
    2. Finding better parking spots for drone launches
    3. Potentially converting some drone deliveries back to truck deliveries
    """
    for truck in solution.trucks:
        if not truck.drones:
            continue
            
        # Calculate current emissions
        truck_route_emissions = calculate_route_metric(
            truck.route, truck_matrix, mapper, truck_cache
        )
        
        drone_emissions = sum(
            calculate_route_metric(drone.route, drone_matrix, mapper, drone_cache)
            for drone in truck.drones
        )
        
        current_total = truck_route_emissions + drone_emissions
        
        # Try to optimize each drone's route
        for drone in truck.drones:
            # Find all feasible parking spots in truck's route
            parking_spots = [node for node in truck.route 
                           if node.node_type == 'Parking Lot']
            
            customer = drone.visit_node
            current_parking = drone.route[0]
            
            best_parking = current_parking
            best_emissions = current_total
            
            # Try each parking spot
            for parking in parking_spots:
                if parking == current_parking:
                    continue
                    
                # Calculate new drone distance
                new_drone_distance = 2 * drone_matrix[mapper[parking.id], mapper[customer.id]]
                
                # Check if within drone's range
                if new_drone_distance <= drone.max_distance:
                    # Calculate total emissions with this change
                    temp_route = [parking, customer, parking]
                    new_drone_emissions = calculate_route_metric(
                        temp_route, drone_matrix, mapper, drone_cache
                    )
                    
                    if new_drone_emissions < best_emissions:
                        best_parking = parking
                        best_emissions = new_drone_emissions
            
            # Apply best change if improvement found
            if best_parking != current_parking:
                # Update parking lot demands
                current_parking._demand -= customer._demand
                best_parking._demand += customer._demand
                
                # Update drone route
                drone.route = [best_parking, customer, best_parking]
                
                # Remove old parking if no longer needed
                if current_parking._demand == 0 and current_parking in truck.route:
                    truck.route.remove(current_parking)            
@perturbations_timer
def swap_interruta_saving(trucks: List[Truck], mapper: Dict[str,int], matrix: np.ndarray, cache: RouteCache) -> None:
    """Swaps nodes between routes based on savings criterion"""
    used_trucks = [truck for truck in trucks if truck.is_used]
    if len(used_trucks) < 2:
        return

    savings = []
    # Calculate all potential savings
    for i, truck_a in enumerate(used_trucks[:-1]):
        for truck_b in used_trucks[i+1:]:
            for pos_a in range(1, len(truck_a.route) - 1):
                for pos_b in range(1, len(truck_b.route) - 1):
                    # Create temporary routes
                    route_a = truck_a.route.copy()
                    route_b = truck_b.route.copy()
                    
                    # Swap nodes
                    node_a = route_a[pos_a]
                    node_b = route_b[pos_b]
                    route_a[pos_a] = node_b
                    route_b[pos_b] = node_a
                    
                    # Calculate savings
                    orig_dist_a = calculate_route_metric(truck_a.route, matrix, mapper, cache)
                    orig_dist_b = calculate_route_metric(truck_b.route, matrix, mapper, cache)
                    new_dist_a = calculate_route_metric(route_a, matrix, mapper, cache)
                    new_dist_b = calculate_route_metric(route_b, matrix, mapper, cache)
                    
                    saving = orig_dist_a + orig_dist_b - new_dist_a - new_dist_b
                    if saving > -0.5:  # Keep original threshold
                        savings.append({
                            'saving': saving,
                            'truck_a': truck_a,
                            'truck_b': truck_b,
                            'node_a': node_a,
                            'node_b': node_b
                        })
    
    # Sort savings by value
    savings.sort(key=lambda x: x['saving'], reverse=True)
    
    # Apply feasible swaps
    for swap in savings:
        truck_a = swap['truck_a']
        truck_b = swap['truck_b']
        node_a = swap['node_a']
        node_b = swap['node_b']
        
        # Check if nodes still exist in routes
        if node_a in truck_a.route and node_b in truck_b.route:
            pos_a = truck_a.route.index(node_a)
            pos_b = truck_b.route.index(node_b)
            
            # Check capacity constraints
            new_cap_a = truck_a.used_capacity - node_a._demand + node_b._demand
            new_cap_b = truck_b.used_capacity - node_b._demand + node_a._demand
            
            if new_cap_a <= truck_a.capacity and new_cap_b <= truck_b.capacity:
                # Apply swap
                truck_a.route[pos_a] = node_b
                truck_b.route[pos_b] = node_a
                truck_a.used_capacity = new_cap_a
                truck_b.used_capacity = new_cap_b
                
                # Optimize new routes
                truck_a.route = twoopt_until_local_optimum(truck_a.route, matrix, mapper, cache)
                truck_b.route = twoopt_until_local_optimum(truck_b.route, matrix, mapper, cache)
                break