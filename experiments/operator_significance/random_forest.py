import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_parquet(r'experiments\operator_significance\operator_performance_data.parquet')

# Operators list (assumed to be defined in your original script)
OPERATORS = [
    'shuffle_route',
    'group_parkings',
    'add_parking',
    'swap_interruta_savings',
    'transfer_node_savings',
    'swap_truck_drone_savings',
    'launch_drone_savings',
    'swap_interruta_random',
    'transfer_node_random']

def prepare_instance_data(instance_df):
    """Prepare data for a specific instance."""
    # One-hot encode operators
    operator_columns = {op: [] for op in OPERATORS}
    for ops in instance_df['active_operators']:
        for op in OPERATORS:
            operator_columns[op].append(1 if op in ops else 0)
    operator_df = pd.DataFrame(operator_columns)
    
    # Combine with target
    full_data = pd.concat([operator_df, instance_df['final_emissions']], axis=1).dropna()
    X = full_data[operator_columns.keys()]
    y = full_data['final_emissions']
    
    return X, y

def train_instance_model(instance, X, y):
    """Train a random forest model for a specific instance."""
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    
    # Evaluate the model
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"Instance {instance} R²: {r2:.4f}")
    print(f"Instance {instance} MSE: {mse:.4f}")
    
    # Get feature importance
    importance = pd.DataFrame({
        'Operator': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    return importance

def plot_importance(instance, importance):
    """Plot operator importance for a specific instance."""
    importance.plot(kind='bar', x='Operator', y='Importance', title=f'Operator Importance for Instance {instance}')
    plt.xlabel('Operator')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

def analyze_instances(df):
    """Analyze each instance separately."""
    instances = df['instance'].unique()
    
    instance_data = {inst: df[df['instance'] == inst] for inst in instances}

    importance_by_instance = {}
    for inst, data in instance_data.items():
        data = data.reset_index(drop=True)
        print(f"Analyzing instance: {inst}")
        X, y = prepare_instance_data(data)
        importance = train_instance_model(inst, X, y)
        plot_importance(inst, importance)
        importance_by_instance[inst] = importance
    
    return importance_by_instance

# Run the analysis
if __name__ == '__main__':
    importance_results = analyze_instances(df)
