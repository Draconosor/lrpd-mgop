from algorithms.construction import construction_timer, initialize_solution
from algorithms.local_search import local_search, ls_timer
from algorithms.perturbations.decorators import perturbations_timer
from utils.io import read_instance
from utils.seed_setter import global_seed_setter

global_seed_setter(0)

instance_nodes, truck_dm, drone_dm, times_truck, times_drone, instance_trucks, instance_drones, mapper = read_instance('C10P15T5D15')

initial_sol = initialize_solution(instance_nodes, mapper, truck_dm, drone_dm, instance_trucks, instance_drones) # type: ignore

parameters = (mapper, truck_dm, drone_dm, times_truck, times_drone)

a, tracker = local_search(parameters, initial_sol) # type: ignore

# Sort by 'total_exec_time' in descending order
#sorted_timing_data = dict(sorted(perturbations_timer.get_stats().items(), key=lambda x: x[1]['total_exec_time'], reverse=True))

# Print the sorted dictionary
#for key, value in sorted_timing_data.items():
#   print(f"{key}: {value}")
    
    


# Retrieve the best solution based on emissions
best_solution = min(a.items(), key=lambda item: item[1][0])
print('CONSTRUCTION TIMER')
print(construction_timer.get_stats())
print('PERTURBATIONS TIMER')
print(perturbations_timer.get_stats())
print('PERTURBATIONS QUALITY TRACKER')
print(tracker.get_stats())