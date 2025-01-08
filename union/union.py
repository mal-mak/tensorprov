import pandas as pd
import torch
import time
import numpy as np
import matplotlib.pyplot as plt


def sparse_tensor_prov(data_tensor_1, data_tensor_2):
    size_1 = len(data_tensor_1)
    size_2 = len(data_tensor_2)
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)
    provenance = torch.zeros((len(combined_df), 2), dtype=torch.int32)

    provenance[:size_1, 0] = 1  # Set '1' for rows from data_tensor_1
    provenance[size_1:, 1] = 1  # Set '1' for rows from data_tensor_2

    return combined_df, provenance


def dense_tensor_prov(data_tensor_1, data_tensor_2):
    size_1 = len(data_tensor_1)
    size_2 = len(data_tensor_2)
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)
    provenance = np.zeros((len(combined_df), 2), dtype=np.int32)

    provenance[:size_1, 0] = 1  # Set '1' for rows from data_tensor_1
    provenance[size_1:, 1] = 1  # Set '1' for rows from data_tensor_2

    return combined_df, provenance


def add_source_identifiers(data_tensor_1, data_tensor_2):
    data_tensor_1 = data_tensor_1.copy()
    data_tensor_2 = data_tensor_2.copy()
    data_tensor_1["source_id"] = "D¹"
    data_tensor_2["source_id"] = "D²"
    return data_tensor_1, data_tensor_2

def append_with_provenance(data_tensor_1, data_tensor_2):
    """
    Append two datasets with provenance (source identifiers).
    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset with source identifiers.
        data_tensor_2 (pd.DataFrame): Second dataset with source identifiers.
    Returns:
        pd.DataFrame: Combined DataFrame with provenance.
    """
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)
    return combined_df


def compare_execution_times(data_tensor_1, data_tensor_2):
    # Track the execution times for sparse and dense methods
    sparse_times = []
    dense_times = []
    
    for i in range(1, 6):  # Perform comparisons with different sizes of datasets
        # Scale data tensors
        scaled_data_1 = data_tensor_1.sample(frac=i / 5, random_state=42)
        scaled_data_2 = data_tensor_2.sample(frac=i / 5, random_state=42)

        # Time sparse tensor operation
        start_time = time.time()
        sparse_tensor_prov(scaled_data_1, scaled_data_2)
        sparse_times.append(time.time() - start_time)

        # Time dense tensor operation
        start_time = time.time()
        dense_tensor_prov(scaled_data_1, scaled_data_2)
        dense_times.append(time.time() - start_time)
    
    return sparse_times, dense_times


# Generate sample data
data_tensor_1 = pd.DataFrame({
    'Feature_A': np.random.randn(100),
    'Feature_B': np.random.randn(100),
})
data_tensor_2 = pd.DataFrame({
    'Feature_A': np.random.randn(100),
    'Feature_B': np.random.randn(100),
})

# Add source identifiers
data_tensor_1, data_tensor_2 = add_source_identifiers(data_tensor_1, data_tensor_2)

# Compare execution times for sparse and dense methods
sparse_times, dense_times = compare_execution_times(data_tensor_1, data_tensor_2)

# Plot execution time comparison
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4, 5], sparse_times, marker='o', label='Sparse Tensor Method', color='blue')
plt.plot([1, 2, 3, 4, 5], dense_times, marker='x', label='Dense Tensor Method', color='red')
plt.xlabel('Scaling Factor (Relative Dataset Size)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison: Sparse vs Dense Tensor Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('execution_time_comparison.png')

# Show the plot
plt.show()
