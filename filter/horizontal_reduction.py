import pandas as pd
import torch


def transform(input_df: pd.DataFrame, filter: str):
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
    try:
        output_df = input_df.query(filter)
        return output_df
    except Exception as e:
        print(
            f'Invalid filter. It should be a valid query condition string. Example: "(col1 > 2) & (col2 < 7)"\nError: {e}'
        )


def provenance(input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True):
    """
    Constructs a tensor that records the provenance of each row in the `output_df` with 
    respect to `input_df`. The provenance tensor indicates which rows in the original 
    DataFrame (`input_df`) are retained in the ransformed DataFrame (`output_df`), 
    based on the applied filter.

    Parameters:
    - input_df (pd.DataFrame): The original input DataFrame containing the unfiltered data.
    - output_df (pd.DataFrame): The filtered output DataFrame containing only the rows
        that match the filter condition.
    - sparse (bool, optional): If True, returns a sparse COO tensor indicating provenance
        information. If False, returns a dense tensor expanded across columns to match
        the shape of `input_df`. Default is True.

    Returns:
    - torch.Tensor: A sparse or dense tensor, depending on the `sparse` flag.
        - If `sparse=True`, a sparse COO tensor with shape `(input_df.shape[0],)`, where
          each non-zero element corresponds to a retained row in `output_df`.
        - If `sparse=False`, a dense tensor with shape `(input_df.shape[0], input_df.shape[1])`
          that has the provenance information duplicated across columns.

    Example:
    >>> data = {"col1": [1, 2, 3, 4, 5], "col2": [8, 6, 9, 7, 10]}
    >>> input_df = pd.DataFrame(data)
    >>> filter_condition = "(col1 > 2) & (col2 > 7)"
    >>> output_df = transform(input_df, filter_condition)
    >>> sparse_tensor = provenance(input_df, output_df, sparse=True)
    >>> sparse_tensor
    tensor(indices=tensor([[2, 4]]),
           values=tensor([1, 1], dtype=torch.int8),
           size=(5,), nnz=2, dtype=torch.int8, layout=torch.sparse_coo)

    >>> dense_tensor = provenance(input_df, output_df, sparse=False)
    >>> dense_tensor
    tensor([[0, 0],
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1]], dtype=torch.int8)
    """

    # Check if the output_df contains columns that exist in input_df
    if not set(output_df.columns).issubset(input_df.columns):
        raise ValueError(
            "Error: One or more columns in the output_df are not found in input_df."
        )

    # Identify retained rows
    retained_rows = output_df.index
    try:
        retained_rows = [input_df.index.get_loc(row) for row in output_df.index]
    except KeyError as e:
        raise KeyError(
            f"Error: One or more rows in output_df are not found in input_df. {e}"
        )

    if sparse:
        try:
            values = torch.ones(len(retained_rows), dtype=torch.int8)
            # Create a COO tensor for provenance
            provenance_tensor = torch.sparse_coo_tensor(
                indices=[retained_rows], values=values, size=(input_df.shape[0],)
            )
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create sparse tensor. {e}")
    else:
        try:
            # Initialize a zero tensor of shape (input_df.shape[0], input_df.shape[1])
            provenance_tensor = torch.zeros(
                input_df.shape[0], input_df.shape[1], dtype=torch.int8
            )
            for row_idx in retained_rows:
                provenance_tensor[row_idx, :] = 1
        except Exception as e:
            raise RuntimeError(f"Error: Failed to create dense tensor. {e}")

    return provenance_tensor
