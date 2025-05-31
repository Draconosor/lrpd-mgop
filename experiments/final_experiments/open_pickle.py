import pickle
import pandas as pd
from utils.io import read_instance
from utils.metrics import RouteCache, calculate_route_metric

nodes, truck_dm, drone_dm, times_truck, times_drone, trucks, drones, mapper = read_instance('C30P15T5D15')

SOLUTION_MEMORY_PATH = "experiments/final_experiments/solution_memories.pkl"

SOLUTION_PATH = "experiments/final_experiments/solutions.pkl"

def load_data_to_list(file_path: str) -> list:
    solutions = []
    try:
        with open(file_path, "rb") as f:
            while True:
                try:
                    solutions.append(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        print("No solution file found.")
    return solutions

# Load solutions
solution_memories = load_data_to_list(SOLUTION_MEMORY_PATH)
solutions = load_data_to_list(SOLUTION_PATH)

joined_memories = pd.concat(solution_memories, ignore_index=True)
joined_memories.to_parquet('experiments/final_experiments/solution_memories.parquet')



















sol = next(solution for solution in solutions if solution.sample_name == 'C10P10T5D15 28')
print('TRUCKS ------------------------')
print([t.route for t in sol.trucks if t.is_used])
print([calculate_route_metric(t.route, truck_dm, mapper, RouteCache()) for t in sol.trucks if t.is_used])
print('DRONES ------------------------')
print([(d.route ,calculate_route_metric(d.route, drone_dm, mapper, RouteCache())) for d in sol.drones if d.is_used])


