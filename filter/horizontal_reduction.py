import pandas as pd
import numpy as np
import torch
import time
import hashlib
import matplotlib.pyplot as plt



def transform(input_df: pd.DataFrame, filter: str) -> pd.DataFrame:
    """
    Filters the rows of a DataFrame based on a specified condition.

    Parameters:
        input_df (pd.DataFrame): The original DataFrame to be filtered.
        filter (str): The condition string used to filter rows. The string must be a valid query condition
                      that can be applied using the `DataFrame.query()` method.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows that satisfy the specified filter condition.

    Example:
        >>> df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        >>> transform(df, "col1 > 1")
           col1  col2
        1     2     5
        2     3     6
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )

    try:
        output_df = input_df.query(filter)
        return output_df
    except Exception as e:
        print(
            f'Invalid filter. It should be a valid query condition string. Example: "(col1 > 2) & (col2 < 7)"\nError: {e}'
        )
        return pd.DataFrame()


# Your existing function
def provenance_index_matching(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
) -> torch.Tensor:
    """
    Create a tensor (provenance) where the (i, j) entry is 1 if the i-th row of
    input_df matches the j-th row of output_df, otherwise 0.

    Parameters:
        - input_df (pd.DataFrame or torch.Tensor): The original DataFrame.
        - output_df (pd.DataFrame or torch.Tensor): The filtered DataFrame.
        - sparse (bool): If True, returns a sparse tensor. Defaults to True.

    Returns:
        - torch.Tensor: A sparse or dense tensor representing the provenance.

    Example:
    >>> input_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    >>> output_data = {"col1": [2, 3], "col2": [5, 6]}
    >>> input_df = pd.DataFrame(input_data)
    >>> output_df = pd.DataFrame(output_data)
    >>> provenance_tensor = provenance_index_matching(input_df, output_df, False)
    >>> provenance_tensor
    tensor([[0, 0],
            [1, 0],
            [0, 1]])
    """
    # Check if the input is a pandas DataFrame, if not, convert it
    if isinstance(input_df, dict):
        input_df = pd.DataFrame(input_df)
    if isinstance(output_df, dict):
        output_df = pd.DataFrame(output_df)

    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )

    # Check if the output_df contains columns that exist in input_df
    if not set(output_df.columns).issubset(input_df.columns):
        raise ValueError(
            "Error: One or more columns in the output_df are not found in input_df."
        )

    # Identify retained rows based on exact content match
    retained_rows = []
    for i, output_row in output_df.iterrows():
        for j, input_row in input_df.iterrows():
            if output_row.equals(input_row):
                retained_rows.append(j)
                break
        else:
            # If no match found, append a placeholder (-1 or any other indication)
            retained_rows.append(-1)

    if sparse:
        try:
            # Create indices for a sparse COO tensor
            indices = torch.tensor(
                [
                    [r for r in retained_rows if r != -1],
                    [i for i, r in enumerate(retained_rows) if r != -1],
                ],
                dtype=torch.int64,
            )
            values = torch.ones(indices.shape[1], dtype=torch.int8)
            # Create the sparse tensor
            provenance_tensor = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=(input_df.shape[0], output_df.shape[0]),
            )
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create sparse tensor. {e}")
    else:
        try:
            # Create a dense tensor and mark retained row relationships
            provenance_tensor = torch.zeros(
                (input_df.shape[0], output_df.shape[0]), dtype=torch.int8
            )
            for output_idx, input_idx in enumerate(retained_rows):
                if input_idx != -1:
                    provenance_tensor[input_idx, output_idx] = 1
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create dense tensor. {e}")

    return provenance_tensor


def hash_row(row) -> str:
    """
    Compute a unique hash for a given row using the SHA256 algorithm.
    """
    row_str = ",".join(map(str, row))
    return hashlib.sha256(row_str.encode()).hexdigest()


def provenance_by_hashing(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
) -> torch.Tensor:
    """
    Create a tensor that represents the provenance of rows between two DataFrames
    based on matching hashes of their rows.

    Parameters:
        input_df (pd.DataFrame or torch.Tensor): The original DataFrame whose rows are to be matched.
        output_df (pd.DataFrame or torch.Tensor): The filtered DataFrame whose rows are to be matched against.
        sparse (bool): If True, returns a sparse tensor. If False, returns a dense tensor.
                        Defaults to True.

    Returns:
        torch.Tensor: A tensor (sparse or dense) representing the provenance. The tensor has
                      shape (n_input_rows, n_output_rows), where each entry (i, j) is 1 if
                      the i-th row of `input_df` matches the j-th row of `output_df`, otherwise 0.

    Example:
        >>> input_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        >>> output_data = {"col1": [2, 3], "col2": [5, 6]}
        >>> input_df = pd.DataFrame(input_data)
        >>> output_df = pd.DataFrame(output_data)
        >>> provenance_tensor = provenance_by_hashing(input_df, output_df, False)
        >>> provenance_tensor
        tensor([[0, 0],
                [1, 0],
                [0, 1]])
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )

    # Check if the output_df contains columns that exist in input_df
    if not set(output_df.columns).issubset(input_df.columns):
        raise ValueError(
            "Error: One or more columns in the output_df are not found in input_df."
        )
    # Compute hashes for rows in both DataFrames
    input_hashes = input_df.apply(hash_row, axis=1).values
    output_hashes = output_df.apply(hash_row, axis=1).values

    # Collect matching indices
    indices = []
    for i, input_hash in enumerate(input_hashes):
        for j, output_hash in enumerate(output_hashes):
            if input_hash == output_hash:
                indices.append((i, j))

    if sparse:
        # Create sparse tensor
        if indices:
            indices_tensor = torch.tensor(
                indices, dtype=torch.long
            ).T  # Transpose for sparse representation
            values = torch.ones(len(indices), dtype=torch.int8)
            sparse_shape = (len(input_hashes), len(output_hashes))
            return torch.sparse_coo_tensor(
                indices_tensor, values, sparse_shape, dtype=torch.int8
            )
        else:
            # Return an empty sparse tensor if no matches are found
            return torch.sparse_coo_tensor(
                size=(len(input_hashes), len(output_hashes)), dtype=torch.int8
            )
    else:
        # Create dense matrix
        provenance_matrix = torch.zeros(
            (len(input_hashes), len(output_hashes)), dtype=torch.int8
        )
        for i, j in indices:
            provenance_matrix[i, j] = 1
        return provenance_matrix


# Performance comparison
def compare(input_df: pd.DataFrame, filter: str) -> None:
    """
    Compare performance and results of sparse and dense provenance methods.

    Parameters:
        - input_df (pd.DataFrame or torch.Tensor): Original dataset (2D tensor or DataFrame).
        - filter (str): Function to filter rows from the input DataFrame.

    Returns:
        - None
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )
    output_df = transform(input_df, filter)

    # Method 1: Index Matching
    print("INDEX MATCHING\n")
    start = time.time()
    sparse_provenance = provenance_index_matching(input_df, output_df, sparse=True)
    sparse_time = time.time() - start
    print(f"Sparse Tensor Time: {sparse_time:.6f}s\n")
    print(f"Provenance Sparse Tensor :\n{sparse_provenance}\n")

    start = time.time()
    dense_provenance = provenance_index_matching(input_df, output_df, sparse=False)
    dense_time = time.time() - start
    print(f"Dense Tensor Time: {dense_time:.6f}s\n")
    print(f"Provenance dense Tensor :\n{dense_provenance}\n")

    # Verify consistency
    consistent = torch.equal(sparse_provenance.to_dense(), dense_provenance)
    print(f"Results Consistent: {consistent}")

    print("\n", "-" * 20, "\n")
    # Method 2: By Hashing
    print("BY HASHING\n")
    start = time.time()
    sparse_provenance = provenance_by_hashing(input_df, output_df, sparse=True)
    sparse_time = time.time() - start
    print(f"Sparse Tensor Time: {sparse_time:.6f}s\n")
    print(f"Provenance Sparse Tensor :\n{sparse_provenance}\n")

    start = time.time()
    dense_provenance = provenance_by_hashing(input_df, output_df, sparse=False)
    dense_time = time.time() - start
    print(f"Dense Tensor Time: {dense_time:.6f}s\n")
    print(f"Provenance dense Tensor :\n{dense_provenance}\n")

    # Verify consistency
    consistent = torch.equal(sparse_provenance.to_dense(), dense_provenance)
    print(f"Results Consistent: {consistent}")


def provenance(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
) -> torch.Tensor | None:
    """
    Compute the provenance between two DataFrames.

    Parameters:
        - input_df (pd.DataFrame or torch.Tensor): The original DataFrame.
        - output_df (pd.DataFrame or torch.Tensor): The filtered or transformed DataFrame.
        - sparse (bool): Whether to return a sparse tensor. Default is True.

    Returns:
        - torch.Tensor or None: The provenance tensor (either sparse or dense), or None if both methods fail.
    """
    try:
        return provenance_index_matching(input_df, output_df, sparse)
    except Exception as e:
        print(f"Index matching failed: {e}. Falling back to provenance_by_hashing.")
        try:
            return provenance_by_hashing(input_df, output_df, sparse)
        except Exception as e:
            # If both methods fail, return None
            print(f"Both methods failed to compute provenance: {e}")
            return None

def main():
    # Generate sample data
    sizes = [100, 1000, 2000, 3000]
    methods = ['Index Matching', 'Hashing']
    sparse_times = {method: [] for method in methods}
    dense_times = {method: [] for method in methods}

    for size in sizes:
        input_df = pd.DataFrame({
            'A': np.random.randint(0, 100, size),
            'B': np.random.randint(0, 100, size),
            'C': np.random.randint(0, 100, size)
        })
        filter_condition = 'A > 50'
        output_df = transform(input_df, filter_condition)

        print(f"\nTesting with {size} rows:")

        # Test Index Matching
        start = time.time()
        provenance_index_matching(input_df, output_df, sparse=True)
        sparse_times['Index Matching'].append(time.time() - start)

        start = time.time()
        provenance_index_matching(input_df, output_df, sparse=False)
        dense_times['Index Matching'].append(time.time() - start)

        # Test Hashing
        start = time.time()
        provenance_by_hashing(input_df, output_df, sparse=True)
        sparse_times['Hashing'].append(time.time() - start)

        start = time.time()
        provenance_by_hashing(input_df, output_df, sparse=False)
        dense_times['Hashing'].append(time.time() - start)

    # Plotting
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^', 'D']
    for i, method in enumerate(methods):
        plt.plot(sizes, sparse_times[method], label=f'{method} (Sparse)', marker=markers[i*2])
        plt.plot(sizes, dense_times[method], label=f'{method} (Dense)', marker=markers[i*2+1], linestyle='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Rows')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison of Horizontal Reduction methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('provenance_performance.png')
    plt.show()

    # Print final results
    print("\nFinal Results:")
    for method in methods:
        print(f"\n{method}:")
        print(f"  Sparse: {sparse_times[method]}")
        print(f"  Dense: {dense_times[method]}")

if __name__ == "__main__":
    main()
