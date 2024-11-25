import pandas as pd
import torch


def transform(input_df: pd.DataFrame, filter: str):
    """
    Applies a `filter` operation to `input_df`, returning a DataFrame with rows
    that meet the specified condition. This function serves as a horizontal reduction
    operation by filtering rows based on the provided condition.

    Parameters:
    - input_df (pd.DataFrame): The original DataFrame to be filtered.
        Must be a Pandas DataFrame containing the input data.
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
    output_df = input_df.query(filter)
    return output_df


def provenance(input_df: pd.DataFrame, output_df: pd.DataFrame, sparse: bool = True):
    """
    Constructs a sparse tensor in COO (Coordinate) format that records the provenance
    of each row in the `output_df` with respect to `input_df`. The provenance tensor
    indicates which rows in the original DataFrame (`input_df`) are retained in the
    transformed DataFrame (`output_df`), based on the applied filter.

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
    # Identify retained rows
    indices = output_df.index
    values = torch.ones(len(indices), dtype=torch.int8)

    # Create a COO tensor for provenance
    provenance_tensor = torch.sparse_coo_tensor(
        indices=[indices], values=values, size=(input_df.shape[0],)
    )

    if sparse:
        return provenance_tensor
    else:
        provenance_tensor_dense = (
            provenance_tensor.to_dense().unsqueeze(1).repeat(1, input_df.shape[1])
        )  # Duplicate the vector across columns to get the right shape
        return provenance_tensor_dense
