# pylint: disable=no-member
import torch
import time
import matplotlib.pyplot as plt
import os

# Function to perform oversampling
def oversample(data, method="horizontal", factor=2):
    if method == "horizontal":
        return data.repeat(1, factor)
    elif method == "vertical":
        return data.repeat(factor, 1)
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

# Provenance determination: Method 1 (Sparse tensor)
def determine_provenance_sparse(original_data, augmented_data, method="horizontal"):
    original_rows, original_cols = original_data.size()
    augmented_rows, augmented_cols = augmented_data.size()

    if method == "horizontal":
        col_map = torch.arange(augmented_cols) % original_cols
        indices = torch.stack([col_map, torch.arange(augmented_cols)])
        size = (original_cols, augmented_cols)
    elif method == "vertical":
        row_map = torch.arange(augmented_rows) % original_rows
        indices = torch.stack([torch.arange(augmented_rows), row_map])
        size = (augmented_rows, original_rows)
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

    values = torch.ones(indices.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=size)

# Provenance determination: Method 2 (Dense tensor)
def determine_provenance_dense(original_data, augmented_data, method="horizontal"):
    original_rows, original_cols = original_data.size()
    augmented_rows, augmented_cols = augmented_data.size()

    if method == "horizontal":
        col_map = torch.arange(augmented_cols) % original_cols
        provenance_tensor = torch.zeros(
            (original_cols, augmented_cols), dtype=torch.float32
        )
        for i, col in enumerate(col_map):
            provenance_tensor[col, i] = 1
    elif method == "vertical":
        row_map = torch.arange(augmented_rows) % original_rows
        provenance_tensor = torch.zeros(
            (augmented_rows, original_rows), dtype=torch.float32
        )
        for i, row in enumerate(row_map):
            provenance_tensor[i, row] = 1
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

    return provenance_tensor

# Performance comparison
def compare_methods(data, method="horizontal", factors=[2, 4, 8]):
    sparse_times = []
    dense_times = []

    for factor in factors:
        # Perform oversampling
        augmented_data = oversample(data, method, factor)

        # Method 1: Sparse tensor
        start = time.time()
        determine_provenance_sparse(data, augmented_data, method)
        sparse_times.append(time.time() - start)

        # Method 2: Dense tensor
        start = time.time()
        determine_provenance_dense(data, augmented_data, method)
        dense_times.append(time.time() - start)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(factors, sparse_times, marker='o', label='Sparse Tensor')
    plt.plot(factors, dense_times, marker='x', label='Dense Tensor')
    plt.xlabel('Oversampling Factor')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time Comparison ({method.capitalize()} Oversampling)')
    plt.legend()
    plt.grid(True)

    # Save the plot with a dynamic filename based on the oversampling method
    save_path = f'{method}_oversampling_comparison.png'
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    # Generate sample data
    data = torch.rand(100, 100)

    # Compare methods for horizontal oversampling
    compare_methods(data, method="horizontal", factors=[2, 4, 8, 16])

    # Compare methods for vertical oversampling
    compare_methods(data, method="vertical", factors=[2, 4, 8, 16])
