import pandas as pd
import torch
import time


def transform(input_df: pd.DataFrame, columns: list[str], retain: bool = True):
    """
    This function allows for either retaining or removing a subset of columns from the
    original DataFrame.

    Parameters:
    - input_df (pd.DataFrame): The original DataFrame from which columns will be selected.
    - columns (list[str]): A list of column names to retain or remove.
    - retain (bool, optional): If True, the columns in the `columns` list are kept.
                               If False, the columns are removed. Defaults to True.

    Returns:
    - pd.DataFrame: A new DataFrame with only the selected columns. If `retain` is True,
                    it contains only the specified columns; if False, the specified columns
                    are dropped.

    Raises:
    - KeyError: If any of the columns provided in the list do not exist in the DataFrame.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    >>> transform(df, ['A', 'C'], retain=True)
       A  C
    0  1  5
    1  2  6

    >>> transform(df, ['B'], retain=False)
       A  C
    0  1  5
    1  2  6
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


def provenance(input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True):
    """
    Constructs a provenance tensor indicating which columns were retained from the original
    DataFrame during vertical reduction.

    The provenance tensor records which columns of the original DataFrame remain in the output
    DataFrame. If sparse is set to True, a sparse tensor is returned. If False, the tensor is
    dense.

    Parameters:
    - input_df (pd.DataFrame): The original DataFrame from which columns were selected.
    - output_df (pd.DataFrame): The DataFrame containing the selected columns (after reduction).
    - sparse (bool, optional): If True, a sparse tensor is returned. If False, the tensor is
                               converted to a dense tensor that is repeated for each row.
                               Defaults to True.

    Returns:
    - torch.Tensor: A sparse or dense tensor, depending on the `sparse` flag.
        - If `sparse=True`, a sparse COO tensor with shape `(input_df.shape[1],)`, where
          each non-zero element corresponds to a retained column in `output_df`.
        - If `sparse=False`, a dense tensor with shape `(input_df.shape[0], input_df.shape[1])`
          that has the provenance information duplicated across rows.

    Raises:
    - ValueError: If the columns in `output_df` are not found in `input_df`.
    - KeyError: If any of the columns in `output_df` are not found in `input_df`.
    - RuntimeError: If there is an issue creating the provenance tensor.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    >>> output_df = transform(df, ['A', 'C'], retain=True)
    >>> provenance(df, output_df, sparse=True)
    tensor(indices=tensor([[0, 2]]), values=tensor([1, 1], dtype=torch.int8), size=(3,))

    >>> provenance(df, output_df, sparse=False)
    tensor([[1, 0, 1],
            [1, 0, 1]])
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

    # Identify retained columns by name and map to column indices
    try:
        retained_columns = [input_df.columns.get_loc(col) for col in output_df.columns]
    except KeyError as e:
        raise KeyError(
            f"Error: One or more column names in output_df are not found in input_df. {e}"
        )

    if sparse:
        try:
            values = torch.ones(len(retained_columns), dtype=torch.int8)
            # Create a COO provenance tensor
            provenance_tensor = torch.sparse_coo_tensor(
                indices=[retained_columns], values=values, size=(input_df.shape[1],)
            )
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create sparse tensor. {e}")
    else:
        try:
            # Initialize a zero tensor of shape (input_df.shape[0], input_df.shape[1])
            provenance_tensor = torch.zeros(
                input_df.shape[0], input_df.shape[1], dtype=torch.int8
            )
            for col_idx in retained_columns:
                provenance_tensor[:, col_idx] = 1
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create dense tensor. {e}")

    return provenance_tensor


# Performance comparison
def compare(input_df, columns, retain):
    """
    Compare performance and results of sparse and dense provenance methods.

    Parameters:
        data (torch.Tensor): Original dataset (2D tensor).
        method (str): Oversampling method ('horizontal' or 'vertical').
        factor (int): Multiplication factor for oversampling.

    Returns:
        None
    """
    if isinstance(input_df, torch.Tensor):
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame(
            input_df.numpy(), columns=[f"col{i}" for i in range(input_df.shape[1])]
        )
    output_df = transform(input_df, columns, retain)

    # Method 1: Sparse tensor
    start = time.time()
    sparse_provenance = provenance(input_df, output_df, sparse=True)
    sparse_time = time.time() - start
    print(f"Sparse Tensor Time: {sparse_time:.6f}s\n")
    print(f"Provenance Sparse Tensor : {sparse_provenance}\n")

    # Method 2: Dense tensor
    start = time.time()
    dense_provenance = provenance(input_df, output_df, sparse=False)
    dense_time = time.time() - start
    print(f"Provenance dense Tensor : {dense_provenance}\n")
    print(f"Dense Tensor Time: {dense_time:.6f}s\n")

    # Verify consistency
    sparse_dense_diff = (
        sparse_provenance.to_dense()
        if sparse_provenance.is_sparse
        else sparse_provenance
    ) - dense_provenance[
        0, :
    ]  # Because dense_provenance is of the same shape as input_df whereas sparse_provenance just represents the retained columns (input_df.shape[1])
    consistent = torch.allclose(sparse_dense_diff, torch.zeros_like(sparse_dense_diff))
    print(f"Results Consistent: {consistent}")
