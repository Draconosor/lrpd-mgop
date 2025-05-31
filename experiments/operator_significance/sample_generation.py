import itertools
import json
import multiprocessing
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from scipy import stats
from tqdm import tqdm

from algorithms.construction import initialize_solution
from algorithms.metaheuristics.local_search import local_search
from utils.io import read_instance
from utils.metrics import RouteCache
from utils.seed_setter import global_seed_setter

# Configuration
INSTANCES = ['C10P15T5D15', 'C20P15T5D15', 'C30P15T5D15']  # Add more instances if needed
N_TRIALS = 30
N_COMBINATIONS = 50  # Number of combinations for fractional factorial design

# Classify operators
SINGLE_OPERATORS = [
    'shuffle_route',
    'group_parkings',
    'add_parking',
    'swap_interruta_savings',
    'transfer_node_savings',
    'swap_truck_drone_savings',
    'launch_drone_savings'
]

PAIRED_OPERATORS = [
    'swap_interruta_random',
    'transfer_node_random',
]

OPERATORS = SINGLE_OPERATORS + PAIRED_OPERATORS

N_PROCESSES = multiprocessing.cpu_count()

def generate_fractional_factorial_design(single_ops: List[str], paired_ops: List[str], n_combinations: int) -> List[List[str]]:
    """Generate a fractional factorial design with the constraint that at least one single operator is active."""
    valid_combinations = []

    # Generate all possible combinations with at least one single operator
    for r in range(1, len(single_ops) + 1):
        for single_combo in itertools.combinations(single_ops, r):
            # Add paired operators if needed
            for p in range(0, len(paired_ops) + 1):
                for paired_combo in itertools.combinations(paired_ops, p):
                    # Convert the tuple to a list
                    valid_combinations.append(list(single_combo) + list(paired_combo))

    # Randomly sample a subset of combinations
    if n_combinations < len(valid_combinations):
        return random.sample(valid_combinations, n_combinations)
    else:
        return valid_combinations

def run_operator_configuration(args: Tuple[str, int, List[str]]) -> Dict:
    """Run a single trial with a specific operator configuration"""
    instance_name, trial_num, active_ops = args
    global_seed_setter(trial_num)
    active_ops += ['fuse_trucks', 'or_opt_improvement']

    try:
        # Initialize problem
        instance_nodes, truck_dm, drone_dm, times_truck, times_drone, trucks, drones, mapper = read_instance(instance_name)
        initial_sol = initialize_solution(instance_nodes, mapper, truck_dm, drone_dm, trucks, drones) # type: ignore
        params = (mapper, truck_dm, drone_dm, times_truck, times_drone, RouteCache(), RouteCache())

        # Determine which operators to exclude
        exclude_ops = [op for op in OPERATORS if op not in active_ops]

        # Run local search with specified exclusions
        solutions, _ = local_search(params, initial_sol, exclude_operators=exclude_ops) # type: ignore

        # Get best solution
        best_emissions = solutions[min(solutions, key=lambda x: solutions[x]['emissions'])]['emissions']

        return {
            'instance': instance_name,
            'trial': trial_num,
            'active_operators': active_ops,
            'final_emissions': best_emissions,
            'status': 'success'
        }
    except Exception as e:
        print(f"Error {instance_name}-{trial_num}-{active_ops}: {str(e)}")
        return {
            'instance': instance_name,
            'trial': trial_num,
            'active_operators': active_ops,
            'final_emissions': np.nan,
            'status': 'failed'
        }

def analyze_operator_performance(df: pd.DataFrame) -> Dict:
    """Analyze operator performance with instance stratification"""
    analysis = {"instance_analysis": {}, "overall_analysis": []}
    operator_pvalues = defaultdict(list)

    # Per-instance analysis
    for instance in df['instance'].unique():
        instance_df = df[df['instance'] == instance]
        instance_results = []

        for operator in OPERATORS:
            present = instance_df['active_operators'].apply(lambda x: operator in x)
            present_ems = instance_df[present]['final_emissions'].dropna()
            absent_ems = instance_df[~present]['final_emissions'].dropna()

            if len(present_ems) < 2 or len(absent_ems) < 2:
                continue  # Skip underpowered comparisons

            _, p = stats.kruskal(present_ems, absent_ems)
            instance_results.append({"operator": operator, "raw_p": p})
            operator_pvalues[operator].append(p)

        # Adjust within-instance p-values
        pvals = [x["raw_p"] for x in instance_results]
        if pvals:
            _, adj_pvals, _, _ = smm.multipletests(pvals, method='bonferroni')
            for i, res in enumerate(instance_results):
                res["adj_p"] = adj_pvals[i]
                res["significant"] = adj_pvals[i] < 0.05

        analysis["instance_analysis"][instance] = sorted(
            instance_results,
            key=lambda x: x["adj_p"] if "adj_p" in x else 1
        )

    # Overall analysis using Fisher's combined probability test
    overall_results = []
    for operator, pvals in operator_pvalues.items():
        if len(pvals) < 1: continue
        _, combined_p = stats.combine_pvalues(pvals, method='pearson')
        overall_results.append({
            "operator": operator,
            "combined_p": combined_p
        })

    # Adjust overall p-values
    pvals = [x["combined_p"] for x in overall_results]
    if pvals:
        _, adj_pvals, _, _ = smm.multipletests(pvals, method='fdr_bh')
        for i, res in enumerate(overall_results):
            res["adj_p"] = adj_pvals[i]
            res["significant"] = adj_pvals[i] < 0.05

    analysis["overall_analysis"] = sorted(
        overall_results,
        key=lambda x: x["adj_p"] if "adj_p" in x else 1
    )

    return analysis

def run_full_experiment() -> pd.DataFrame:
    """Run the full experiment with all operator configurations"""
    configurations = generate_fractional_factorial_design(SINGLE_OPERATORS, PAIRED_OPERATORS, n_combinations=70)
    tasks = [(inst, trial, config) 
            for inst in INSTANCES
            for trial in range(N_TRIALS)
            for config in configurations]
    
    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_operator_configuration, tasks),
            total=len(tasks),
            desc="Running full experiment",
            unit="trial"
        ))
    
    return pd.DataFrame(results)

class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)

if __name__ == '__main__':
    # Run experiment
    print("🚀 Starting full operator performance experiment...")
    df = run_full_experiment()
    df.to_parquet(r'experiments\operator_significance\operator_performance_data.parquet', index=False)

    # Analyze results
    print("📊 Analyzing operator performance...")
    analysis = analyze_operator_performance(df)
    with open(r'experiments\operator_significance\operator_performance_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, cls=CustomJSONizer)

    print("✅ Experiment complete! Files created:")
    print("- operator_performance_data.parquet (raw data)")
    print("- operator_performance_analysis.json (statistical results)")