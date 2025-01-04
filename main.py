from algorithms.construction import initialize_solution
from algorithms.local_search import local_search, ls_timer
from algorithms.perturbations import perturbations_timer
from utils.io import read_instance

instance_nodes, truck_dm, drone_dm, times_truck, times_drone, instance_trucks, instance_drones, mapper = read_instance('C10P5T5D15')

initial_sol = initialize_solution(instance_nodes, mapper, truck_dm, drone_dm, instance_trucks, instance_drones)

parameters = (mapper, truck_dm, drone_dm, times_truck, times_drone)

a = local_search(parameters, initial_sol)
print(perturbations_timer.get_stats())
print(ls_timer.get_stats())
