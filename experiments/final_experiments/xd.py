from algorithms.construction import initialize_solution
from algorithms.metaheuristics.local_search import local_search, ls_timer
from utils.io import read_instance
from utils.metrics import RouteCache

instance_nodes, truck_dm, drone_dm, times_truck, times_drone, instance_trucks, instance_drones, mapper = read_instance('C10P15T5D15')

initial_sol = initialize_solution(instance_nodes, mapper, truck_dm, drone_dm, instance_trucks, instance_drones)

parameters = (mapper, truck_dm, drone_dm, times_truck, times_drone, RouteCache(), RouteCache())

alfa = 0.008

a, _ = local_search(parameters, initial_sol, n_iter = 200, n_size = 20, specify_operators=['shuffle_route', 'add_parking', 'fuse_trucks', 'group_parkings', 'launch_drone_savings', 'or_opt_improvement'], alfa = alfa)

print(a[min(a, key=lambda x: a[x]['objective'])]['emissions'])
print(a[min(a, key=lambda x: a[x]['objective'])]['makespan_per_truck'])
