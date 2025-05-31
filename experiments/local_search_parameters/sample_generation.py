import itertools
import multiprocessing
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from algorithms.construction import initialize_solution
from algorithms.metaheuristics.local_search import local_search
from utils.io import read_instance
from utils.metrics import RouteCache
from utils.seed_setter import global_seed_setter

# Configuration
INSTANCES = ['C10P15T5D15', 'C20P15T5D15', 'C30P15T5D15']  # Add more instances if needed
N_TRIALS = 50  # Number of trials per combination
N_SIZE_LEVES = [20, 50]
N_ITER_LEVELS = [100, 200, 300]

N_PROCESSES = multiprocessing.cpu_count()

def generate_full_factorial_design() -> List[Tuple[str, int, int]]:
    """Generate all possible combinations of parameters (excluding trials)."""
    combinations = list(itertools.product(INSTANCES, N_SIZE_LEVES, N_ITER_LEVELS))
    return combinations

def run_operator_configuration(args: Tuple[str, int, int, int]) -> Dict:
    """Run a single trial with a specific parameter configuration."""
    instance_name, n_size, n_iter, trial_num = args
    global_seed_setter(trial_num)

    try:
        # Initialize problem
        instance_nodes, truck_dm, drone_dm, times_truck, times_drone, trucks, drones, mapper = read_instance(instance_name)
        initial_sol = initialize_solution(instance_nodes, mapper, truck_dm, drone_dm, trucks, drones) # type: ignore
        params = (mapper, truck_dm, drone_dm, times_truck, times_drone, RouteCache(), RouteCache())

        # Run local search with specified parameters
        solutions, _ = local_search(params, initial_sol, n_size=n_size, n_iter=n_iter) # type: ignore

        # Get best solution
        best_emissions = solutions[min(solutions, key=lambda x: solutions[x]['emissions'])]['emissions']

        return {
            'instance': instance_name,
            'trial': trial_num,
            'n_size': n_size,
            'n_iter': n_iter,
            'final_emissions': best_emissions,
            'status': 'success'
        }
    except Exception as e:
        print(f"Error {instance_name}-{trial_num}-{n_size}-{n_iter}: {str(e)}")
        return {
            'instance': instance_name,
            'trial': trial_num,
            'n_size': n_size,
            'n_iter': n_iter,
            'final_emissions': np.nan,
            'status': 'failed'
        }

def run_full_experiment() -> pd.DataFrame:
    """Run the full experiment with all parameter configurations, repeating each N_TRIALS times."""
    configurations = generate_full_factorial_design()
    
    # Create tasks by repeating each configuration N_TRIALS times
    tasks = [(inst, n_size, n_iter, trial) 
             for inst, n_size, n_iter in configurations 
             for trial in range(N_TRIALS)]
    
    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_operator_configuration, tasks),
            total=len(tasks),
            desc="Running full experiment",
            unit="trial"
        ))
    
    return pd.DataFrame(results)

if __name__ == '__main__':
    # Run experiment
    print("🚀 Starting full parameter performance experiment...")
    df = run_full_experiment()
    df.to_parquet(r'experiments\local_search_parameters\experimental_results.parquet', index=False)