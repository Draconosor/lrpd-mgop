import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.nodes import Node
from models.vehicles import Drone, Truck
from utils.metrics import get_index_mapper


def read_instance_file(instance_id: str) -> pd.ExcelFile:
    """Reads the instance Excel file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    file_path = os.path.join(project_root, "instances", instance_id, f"{instance_id}.xlsx")
    return pd.ExcelFile(file_path)

def create_nodes(instance_file: pd.ExcelFile) -> List[Node]:
    """Creates nodes from the instance data."""
    coords_df = pd.read_excel(instance_file, sheet_name="COORDS")
    demand_df = pd.read_excel(instance_file, sheet_name="DEMANDA")
    
    # Optimize merge operation
    nodes_df = coords_df.merge(demand_df, left_on="NODES", right_on="NODO", how='left')
    nodes_df["NODES"] = nodes_df["NODES"].astype(str)
    
    # Use list comprehension instead of apply
    return [
        Node(
            str(row["NODES"]),
            "Customer" if "Customer" in row["NODES"] 
            else "Parking Lot" if "Depot" in row["NODES"] 
            else "0",
            (row["X"], row["Y"]),
            row["DEMANDA"]
        )
        for _, row in nodes_df.iterrows()
    ]

def create_vehicles(instance_file: pd.ExcelFile) -> Tuple[List[Truck], List[Drone], float, float]:
    """Creates trucks and drones from the instance parameters."""
    # Optimize DataFrame operations
    parameters = pd.read_excel(
        instance_file, 
        sheet_name="PARAMETROS", 
        usecols="E:N"
    ).dropna().T
    
    params = parameters.rename(columns={0: "value"}).to_dict()["value"]
    
    # Create vehicles using list comprehensions
    trucks = [
        Truck(f"T{n}", params["QTR"], params["ETR"]) 
        for n in range(1, int(params["NTRUCKS"]) + 1)
    ]
    
    drones = [
        Drone(
            f"D{n}", 
            params["QDR"], 
            params["EDR"], 
            params["MAXDDR"]*params["EDR"], 
            params["WDR"]
        )
        for n in range(1, int(params["NDRONES"]) + 1)
    ]
    
    return trucks, drones, params["ETR"], params["EDR"]

def read_distance_matrices(instance_file: pd.ExcelFile, etr: float, edr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Optimized reading of distance and time matrices."""
    def process_matrix(sheet_name: str, multiplier: float = 1.0) -> pd.DataFrame:
        df = pd.read_excel(instance_file, sheet_name=sheet_name, index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df = df[df.index] ## Ensure 1:1 ordered matrix
        if multiplier != 1.0:
            df *= multiplier
        return df
    
    # Process all matrices at once
    truck_dm = process_matrix("MANHATTAN", etr)
    drone_dm = process_matrix("EUCLI", edr)
    times_truck = process_matrix("TIEMPOS_CAM")
    times_drone = process_matrix("TIEMPOS_DRON")
    
    return truck_dm.to_numpy(), drone_dm.to_numpy(), times_truck.to_numpy(), times_drone.to_numpy(), get_index_mapper(truck_dm)

def read_instance(instance_id: str) -> Tuple[List[Node], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Truck], List[Drone], Dict[str, int]]:
    """Optimized instance reading function."""
    instance_file = read_instance_file(instance_id)
    
    # Parallel processing could be implemented here for larger instances
    instance_nodes = create_nodes(instance_file)
    instance_trucks, instance_drones, etr, edr = create_vehicles(instance_file)
    truck_dm, drone_dm, times_truck, times_drone, mapper = read_distance_matrices(instance_file, etr, edr)
    
    return instance_nodes, truck_dm, drone_dm, times_truck, times_drone, instance_trucks, instance_drones, mapper