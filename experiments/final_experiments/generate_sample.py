from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from time import perf_counter
import pickle

from algorithms.construction import initialize_solution
from algorithms.metaheuristics.local_search import ObjectiveTracker, local_search
from models.solution import Solution
from utils.io import read_instance
from utils.metrics import RouteCache

CONFIG = {
    'instances': [f.name for f in Path(r'instances').iterdir() if f.is_dir()],
    'runs': 50,
    'operators': ['shuffle_route', 'add_parking', 'fuse_trucks', 'group_parkings', 'launch_drone_savings', 'or_opt_improvement'],
    'n_iter': 200,
    'n_size': 20,
}

SOLUTION_PATH = Path("experiments/final_experiments/solutions.pkl")
SOLUTION_MEMORIES = Path("experiments/final_experiments/solution_memories.pkl")

@dataclass(slots=True)
class LocalSearchResults:
    memory: Dict[Solution, Dict]
    trackers: Tuple[ObjectiveTracker, ObjectiveTracker]
    name : str = ''
    
    @property
    def best_solution(self) -> Tuple[Solution, Dict]:
        min_sol = min(self.memory, key=lambda x: self.memory[x]['objective'])
        min_value = self.memory[min_sol]
        return min_sol, min_value
    
    def to_dataframe(self, instance: str, run: int) -> pd.DataFrame:
        data = {
            'instance': [instance for _ in range(len(self.memory))], 
            'run': [run for _ in range(len(self.memory))], 
            'id': [solution.id for solution in self.memory.keys()], 
            'objective': [self.memory[solution]['objective'] for solution in self.memory.keys()]
            }
        return pd.DataFrame(data)

def process_run(args):
    """Procesa una sola ejecución (1 run para 1 instancia)."""
    instance, run, config = args

    truck_cache = RouteCache()
    drone_cache = RouteCache()

    nodes, truck_dm, drone_dm, times_truck, times_drone, trucks, drones, mapper = read_instance(instance)

    start = perf_counter()
    initial_solution = initialize_solution(nodes, mapper, truck_dm, drone_dm, trucks, drones)  # type: ignore
    construction_time = perf_counter() - start

    start = perf_counter()
    parameters = (mapper, truck_dm, drone_dm, times_truck, times_drone, truck_cache, drone_cache)
    solution_memory, trackers = local_search(parameters, initial_solution, n_iter=config['n_iter'], n_size=config['n_size'], specify_operators=config['operators'], alfa = 1)  # type: ignore
    local_search_time = perf_counter() - start

    results_object = LocalSearchResults(solution_memory, trackers)
    makespan_per_truck = results_object.best_solution[1]['makespan_per_truck']
    best_solution = results_object.best_solution[0] 
    best_solution.sample_name = f"{instance} {run}"  
    results_object.name = f"{instance} {run}"

    # Save solution object to pickle file
    with open(SOLUTION_PATH, "ab") as f:
        pickle.dump(best_solution, f)

    # Save solution object to pickle file
    with open(SOLUTION_MEMORIES, "ab") as f:
        pickle.dump(results_object.to_dataframe(instance, run), f)


    return {
        'instance': instance,
        'run': run,
        'objective': results_object.best_solution[1]['objective'],
        'emissions': results_object.best_solution[1]['emissions'],
        'min_makespan': min(makespan_per_truck.values()),
        'max_makespan': max(makespan_per_truck.values()),
        'used_trucks': len(makespan_per_truck),
        'used_drones': len([drone for drone in results_object.best_solution[0].drones if drone.is_used]),
        'used_parkings': sum(
            1 for lot in results_object.best_solution[0].nodes
            if lot.node_type == 'Parking Lot' and
            any(lot in drone.route for drone in results_object.best_solution[0].drones)
        ),
        'construction_time': construction_time,
        'local_search_time': local_search_time,
        'total_time': construction_time + local_search_time
    }

def run_sampling_parallel(config: Dict) -> pd.DataFrame:
    tasks = [(instance, run, config) for instance in config['instances'] for run in range(config['runs'])]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_results = list(tqdm(pool.imap_unordered(process_run, tasks), total=len(tasks), desc="Progress"))

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df = run_sampling_parallel(CONFIG)
    df.to_parquet(r'experiments\final_experiments\results.parquet')
