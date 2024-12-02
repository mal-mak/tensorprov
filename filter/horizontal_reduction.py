import pandas as pd
import torch
import time
import hashlib


def transform(input_df: pd.DataFrame, filter: str) -> pd.DataFrame:
    """
    Applies a `filter` operation to `input_df`, returning a DataFrame with rows
    that meet the specified condition. This function serves as a horizontal reduction
    operation by filtering rows based on the provided condition.

    Parameters:
        - input_df (pd.DataFrame): The original DataFrame to be filtered.
        - filter (str): The filter condition as a string. This should be a valid query
            condition string, formatted in the style supported by `DataFrame.query()`.
            For example, "col1 > 2 & col2 < 10".

    Returns:
        - pd.DataFrame: A DataFrame containing only rows that satisfy the specified filter condition.

    Example:
    >>> data = {"col1": [1, 2, 3, 4, 5], "col2": [8, 6, 9, 7, 10]}
    >>> input_df = pd.DataFrame(data)
    >>> filter_condition = "(col1 > 2) & (col2 > 7)"
    >>> output_df = transform(input_df, filter_condition)
    >>> output_df
       col1  col2
    2     3     9
    4     5    10
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


def provenance_index_matching(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
) -> torch.Tensor:
    """
    Create a tensor where the (i, j) entry is 1 if the hash of the i-th row of
    input_df matches the hash of the j-th row of output_df, otherwise 0.

    Parameters:
        - input_df (pd.DataFrame): The original DataFrame.
        - output_df (pd.DataFrame): The filtered DataFrame.
        - sparse (bool): If True, returns a sparse tensor. Defaults to True.

    Returns:
        - torch.Tensor: A sparse or dense tensor representing the provenance.
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

    # Identify retained rows
    try:
        retained_rows = [input_df.index.get_loc(row) for row in output_df.index]
    except KeyError as e:
        raise KeyError(
            f"Error: One or more rows in output_df are not found in input_df. {e}"
        )

    if sparse:
        try:
            # Create indices for a sparse COO tensor
            indices = torch.tensor(
                [retained_rows, list(range(len(retained_rows)))], dtype=torch.int64
            )
            values = torch.ones(len(retained_rows), dtype=torch.int8)
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
                provenance_tensor[input_idx, output_idx] = 1
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create dense tensor. {e}")

    return provenance_tensor


def hash_row(row) -> str:
    """
    Compute a unique hash for a given row using SHA256.
    """
    row_str = ",".join(map(str, row))
    return hashlib.sha256(row_str.encode()).hexdigest()


def provenance_by_hashing(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
) -> torch.Tensor:
    """
    Create a tensor where the (i, j) entry is 1 if the hash of the i-th row of
    input_df matches the hash of the j-th row of output_df, otherwise 0.

    Parameters:
        - input_df (pd.DataFrame): The original DataFrame.
        - output_df (pd.DataFrame): The filtered DataFrame.
        - sparse (bool): If True, returns a sparse tensor. Defaults to True.

    Returns:
        - torch.Tensor: A sparse or dense tensor representing the provenance.
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
        input_df (pd.DataFrame or torch.Tensor): Original dataset (2D tensor or DataFrame).
        filter (callable): Function to filter rows from the input DataFrame.

    Returns:
        None
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
    Compute the provenance between two DataFrames using different methods.
    Tries the index matching method first, and falls back to hashing if there's an error.
    If both methods fail, raises a custom ProvenanceError.

    Parameters:
        - input_df (pd.DataFrame): The original DataFrame.
        - output_df (pd.DataFrame): The filtered or transformed DataFrame.
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
