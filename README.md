# Routing Optimization with Trucks and Drones

A Python implementation of a hybrid vehicle routing optimization system that combines traditional truck delivery with drone support. This solution uses various heuristic approaches to minimize emissions and optimize delivery routes.

## Features

- Multi-vehicle routing optimization
- Drone-truck coordination
- Multiple optimization strategies:
  - Nearest Neighbor algorithm for initial route construction
  - 2-opt local search improvement
  - Inter-route node swapping
  - Drone assignment optimization
  - Parking lot optimization
  - Route clustering

## Dependencies

- Python 3.x
- pandas
- numpy
- random
- time
- typing
- dataclasses
- collections

## Core Components

### Classes

- `Nodes`: Represents delivery points, parking lots, and the depot
- `Vehicle`: Base class for transportation units
- `Truck`: Extends Vehicle class with drone management capabilities
- `Drone`: Extends Vehicle class with specific drone attributes
- `Solution`: Represents a complete routing solution
- `Saving`: Handles saving calculations for route optimization
- `RouteCache`: Caches route metrics for performance optimization
- `PerturbationTimer`: Tracks execution times of optimization operations

### Key Functions

1. **Initialization**
   - `read_instance`: Loads problem instance data
   - `create_nodes`: Generates node objects from instance data
   - `create_vehicles`: Creates truck and drone objects

2. **Route Construction**
   - `nn_bigroute`: Implements Nearest Neighbor algorithm
   - `cluster_bigroute`: Splits routes among available trucks
   - `assign_drones_bigroute`: Initial drone assignment

3. **Optimization**
   - `twoopt_until_local_optimum`: Implements 2-opt improvement
   - `swap_interruta_random`: Random inter-route node swapping
   - `swap_interruta_saving`: Savings-based inter-route optimization
   - `optimize_drone_assignments`: Improves drone delivery patterns

## Optimization Process

1. **Initial Solution Construction**
   - Creates a giant route using Nearest Neighbor
   - Assigns drones to suitable deliveries
   - Clusters routes among available trucks

2. **Local Search**
   - Generates neighbors using various perturbation operators
   - Evaluates solutions based on emissions
   - Maintains solution memory to avoid cycling
   - Implements multiple local search strategies

## Performance Monitoring

The `PerturbationTimer` decorator tracks:
- Average execution time per operation
- Minimum and maximum execution times
- Total number of calls
- Total execution time

# Setting Up the Python Environment

Follow the steps below to replicate the Python environment required to run this project.

## Prerequisites

Ensure the following are installed on your system:
- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/) (comes pre-installed with Python 3.4+)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/) or the built-in `venv` module for Python

## Steps to Set Up the Environment

1. **Clone the Repository**  
   Start by cloning this repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**  
   It is recommended to create a virtual environment to isolate project dependencies:
   ```bash
   # For Linux/MacOS
   python3 -m venv .venv

   # For Windows
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**  
   Activate the virtual environment with the following command:
   ```bash
   # For Linux/MacOS
   source .venv/bin/activate

   # For Windows (Command Prompt)
   .venv\Scripts\activate

   # For Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

4. **Install Project Dependencies & Packages**  
   With the virtual environment activated, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
5. **Run the Project**  
   To verify that the setup was successful, run the main script:
   ```bash
   python main.py
   ```

## Notes

- All dependencies for this project are managed in the `requirements.txt` file.
- The folder structure of the project includes:
  - `algorithms/`: Contains algorithm-related logic.
  - `models/`: Stores model files.
  - `instances/`: Holds instances or sample data for testing.
  - `utils/`: Includes utility scripts.
  - `main.py`: Entry point for running the project.
  - `.gitignore`: Specifies files to be ignored by Git.
  - `setup.py`: Script for packaging the project.

## Notes

- The script uses random seed initialization for reproducibility
- Cache mechanisms are implemented to improve performance
- Solution quality is measured primarily by emissions
- Makespan is calculated per truck as a secondary metric, trying to guarantee balanced completion times between trucks.

## About the author

I am Carlos Steven Rodriguez-Salcedo, an MSc candidate in Operations Management at Universidad de La Sabana.

I am an enthusiast of the transformation of logistics processes through technology. I have experience in operations research from academia, data analytics and predictive models applied to the supply chain.