import pandas as pd
import torch
import time
import hashlib


def transform(input_df: pd.DataFrame, columns: list[str], retain: bool = True):
    """
    Filters `input_df`, returning a DataFrame with specified columns only.

    Parameters:
        - input_df (pd.DataFrame): The original DataFrame to be filtered.
        - columns (List[str]): Columns to filter.
        - retain (bool): If True, 'columns' are kept in the output. Else 'columns' are removed.
                        Defaults to True.

    Returns:
        - pd.DataFrame: A DataFrame containing only rows that satisfy the specified filter condition.

    Example:
    >>> data = {"col1": [1, 2, 3, 4, 5], "col2": [8, 6, 9, 7, 10]}
    >>> input_df = pd.DataFrame(data)
    >>> columns = ["(col1"]
    >>> output_df = transform(input_df, columns, True)
    >>> output_df
       col1
    0     1
    1     2
    2     3
    3     4
    4     5
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )

    try:
        if retain:
            output_df = input_df[columns]
            return output_df
        else:
            output_df = input_df.drop(columns=columns)
            return output_df
    except KeyError as e:
        print(
            f"Error: One or more column names not found. Please provide valid column names.\n{e}"
        )


def provenance_column_matching(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
):
    """
    Compute the provenance for vertical reduction by columns matching.

    Parameters:
        input_df (pd.DataFrame): The original DataFrame.
        output_df (pd.DataFrame): The transformed DataFrame with retained columns.
        sparse (bool, optional): If True, returns a sparse tensor. If False, returns a dense tensor. Defaults to True.

    Returns:
        torch.Tensor: A tensor representing the provenance of retained columns from input_df to output_df.

    Raises:
        ValueError: If any columns in output_df are not found in input_df.
        KeyError: If a column in output_df cannot be found in input_df.

    Example:
        >>> input_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        >>> output_df = input_df[['A', 'C']]
        >>> provenance_column_matching(input_df, output_df)
        tensor(indices=tensor([[0, 2]]),
                values=tensor([1, 1]),
                size=(3,), nnz=2, dtype=torch.int8, layout=torch.sparse_coo)
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

    # Identify retained columns
    try:
        retained_columns = [input_df.columns.get_loc(col) for col in output_df.columns]
    except KeyError as e:
        raise KeyError(
            f"Error: One or more column names in output_df are not found in input_df. {e}"
        )

    if sparse:
        try:
            values = torch.ones(len(retained_columns), dtype=torch.int8)
            # Create a sparse tensor for provenance
            provenance_tensor = torch.sparse_coo_tensor(
                indices=[retained_columns], values=values, size=(input_df.shape[1],)
            )
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create sparse tensor. {e}")
    else:
        try:
            # Initialize a dense tensor with zeros
            provenance_tensor = torch.zeros(input_df.shape[1], dtype=torch.int8)
            # Set the retained columns to 1
            for col_idx in retained_columns:
                provenance_tensor[col_idx] = 1
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create dense tensor. {e}")

    return provenance_tensor


def hash_column(column) -> str:
    """
    Compute a unique SHA256 hash for a given column.
    """
    col_str = ",".join(map(str, column))
    return hashlib.sha256(col_str.encode()).hexdigest()


def provenance_by_hashing(
    input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True
):
    """
    Compute the provenance of columns by comparing column-wise hashes between input_df and output_df.

    Parameters:
        input_df (pd.DataFrame): The original DataFrame.
        output_df (pd.DataFrame): The transformed DataFrame with retained columns.
        sparse (bool, optional): If True, returns a sparse tensor. If False, returns a dense tensor. Defaults to True.

    Returns:
        torch.Tensor: A tensor representing the provenance of the retained columns from input_df to output_df.

    Raises:
        ValueError: If any columns in output_df are not found in input_df.
        KeyError: If a column in output_df cannot be found in input_df.

    Example:
        >>> input_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        >>> output_df = input_df[['A', 'C']]
        >>> provenance_by_hashing(input_df, output_df)
        tensor(indices=tensor([[0, 2]]),
                values=tensor([1, 1]),
                size=(3,), nnz=2, dtype=torch.int8, layout=torch.sparse_coo)
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

    # Hash columns in the output DataFrame
    output_hashes = output_df.apply(hash_column, axis=0)

    # Identify the retained columns based on matching hashes
    retained_columns = [
        col
        for col in input_df.columns
        if hash_column(input_df[col]) in output_hashes.values
    ]

    indices = [input_df.columns.get_loc(col) for col in retained_columns]

    if sparse:
        # Create sparse tensor
        values = torch.ones(len(indices), dtype=torch.int8)
        return torch.sparse_coo_tensor(
            indices=[indices], values=values, size=(input_df.shape[1],)
        )
    else:
        # Create dense tensor
        provenance_tensor = torch.zeros(input_df.shape[1], dtype=torch.int8)
        for col_idx in indices:
            provenance_tensor[col_idx] = 1
        return provenance_tensor


# Performance comparison
def compare(input_df, columns, retain):
    """
    Compare the performance and results of different provenance methods.

    Parameters:
        input_df (pd.DataFrame): The original DataFrame.
        columns (list[str]): The columns to retain or remove in the output DataFrame.
        retain (bool): If True, the specified columns are retained; if False, they are removed.

    Returns:
        None: This function prints the results of the provenance comparisons and performance timings.
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )
    output_df = transform(input_df, columns, retain)

    # Method 1: Index Matching
    print("INDEX MATCHING\n")
    start = time.time()
    sparse_provenance = provenance_column_matching(input_df, output_df, sparse=True)
    sparse_time = time.time() - start
    print(f"Sparse Tensor Time: {sparse_time:.6f}s\n")
    print(f"Provenance Sparse Tensor :\n{sparse_provenance}\n")

    start = time.time()
    dense_provenance = provenance_column_matching(input_df, output_df, sparse=False)
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
        return provenance_column_matching(input_df, output_df, sparse)
    except Exception as e:
        print(f"Index matching failed: {e}. Falling back to provenance_by_hashing.")
        try:
            return provenance_by_hashing(input_df, output_df, sparse)
        except Exception as e:
            # If both methods fail, return None
            print(f"Both methods failed to compute provenance: {e}")
            return None
