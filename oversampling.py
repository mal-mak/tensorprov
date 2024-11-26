"Oversampling tensors"
# pylint: disable=no-member
import torch
import time

# Function to perform oversampling
def oversample(data, method='horizontal', factor=2):
    """
    Perform horizontal or vertical oversampling.

    Parameters:
        data (torch.Tensor): Input dataset (2D tensor).
        method (str): Oversampling direction ('horizontal' or 'vertical').
        factor (int): Multiplication factor for oversampling.

    Returns:
        torch.Tensor: Oversampled dataset.
    """
    if method == 'horizontal':
        return data.repeat(1, factor)
    elif method == 'vertical':
        return data.repeat(factor, 1)
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

# Provenance determination: Method 1 (Sparse tensor)
def determine_provenance_sparse(original_data, augmented_data, method='horizontal'):
    """
    Determine provenance using a sparse tensor approach.

    Parameters:
        original_data (torch.Tensor): Original dataset.
        augmented_data (torch.Tensor): Oversampled dataset.
        method (str): Oversampling method ('horizontal' or 'vertical').

    Returns:
        torch.sparse.Tensor: Sparse binary tensor capturing provenance.
    """
    original_rows, original_cols = original_data.size()
    augmented_rows, augmented_cols = augmented_data.size()

    if method == 'horizontal':
        col_map = torch.arange(augmented_cols) % original_cols
        indices = torch.stack([col_map, torch.arange(augmented_cols)])
        size = (original_cols, augmented_cols)
    elif method == 'vertical':
        row_map = torch.arange(augmented_rows) % original_rows
        indices = torch.stack([torch.arange(augmented_rows), row_map])
        size = (augmented_rows, original_rows)
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

    values = torch.ones(indices.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=size)

# Provenance determination: Method 2 (Dense tensor)
def determine_provenance_dense(original_data, augmented_data, method='horizontal'):
    """
    Determine provenance using a dense approach.

    Parameters:
        original_data (torch.Tensor): Original dataset.
        augmented_data (torch.Tensor): Oversampled dataset.
        method (str): Oversampling method ('horizontal' or 'vertical').

    Returns:
        torch.Tensor: Dense binary tensor capturing provenance.
    """
    original_rows, original_cols = original_data.size()
    augmented_rows, augmented_cols = augmented_data.size()

    if method == 'horizontal':
        col_map = torch.arange(augmented_cols) % original_cols
        provenance_tensor = torch.zeros((original_cols, augmented_cols), dtype=torch.float32)
        for i, col in enumerate(col_map):
            provenance_tensor[col, i] = 1
    elif method == 'vertical':
        row_map = torch.arange(augmented_rows) % original_rows
        provenance_tensor = torch.zeros((augmented_rows, original_rows), dtype=torch.float32)
        for i, row in enumerate(row_map):
            provenance_tensor[i, row] = 1
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")

    return provenance_tensor

# Performance comparison
def compare_methods(data, method='horizontal', factor=2):
    """
    Compare performance and results of sparse and dense provenance methods.

    Parameters:
        data (torch.Tensor): Original dataset (2D tensor).
        method (str): Oversampling method ('horizontal' or 'vertical').
        factor (int): Multiplication factor for oversampling.

    Returns:
        None
    """
    print(f"\nEvaluating {method} oversampling with factor {factor}:")

    # Perform oversampling
    augmented_data = oversample(data, method, factor)

    # Method 1: Sparse tensor
    start = time.time()
    sparse_provenance = determine_provenance_sparse(data, augmented_data, method)
    sparse_time = time.time() - start
    print(f"Sparse Tensor Time: {sparse_time:.6f}s")
    print(f"Provenance Sparse Tensor : {sparse_provenance}")

    # Method 2: Dense tensor
    start = time.time()
    dense_provenance = determine_provenance_dense(data, augmented_data, method)
    dense_time = time.time() - start
    print(f"Provenance dense Tensor : {dense_provenance}")
    print(f"Dense Tensor Time: {dense_time:.6f}s")

    # Verify consistency
    sparse_dense_diff = (
        sparse_provenance.to_dense() if sparse_provenance.is_sparse else sparse_provenance
        ) - dense_provenance
    consistent = torch.allclose(sparse_dense_diff, torch.zeros_like(sparse_dense_diff))
    print(f"Results Consistent: {consistent}")
