
import pandas as pd 
import torch


def sparse_tensor_prov(data_tensor_1, data_tensor_2):
    """
    Combine two datasets by appending rows, generate a binary provenance matrix,
    and validate input DataFrames.

    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset.
        data_tensor_2 (pd.DataFrame): Second dataset.

    Returns:
        combined_df (pd.DataFrame): Combined DataFrame.
        provenance (torch.Tensor): Binary provenance matrix.
    """
    # Validate the dataframes
    if not isinstance(data_tensor_1, pd.DataFrame):
        raise TypeError("data_tensor_1 must be a Pandas DataFrame.")
    if not isinstance(data_tensor_2, pd.DataFrame):
        raise TypeError("data_tensor_2 must be a Pandas DataFrame.")
    if data_tensor_1.empty:
        raise ValueError("data_tensor_1 is empty. Provide a non-empty DataFrame.")
    if data_tensor_2.empty:
        raise ValueError("data_tensor_2 is empty. Provide a non-empty DataFrame.")

    # Combine the dataframes
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)

    # Generate provenance matrix
    size_1 = len(data_tensor_1)
    size_2 = len(data_tensor_2)
    provenance = torch.zeros((len(combined_df), 2), dtype=torch.int32)

    provenance[:size_1, 0] = 1  # Set '1' for rows from data_tensor_1
    provenance[size_1:, 1] = 1  # Set '1' for rows from data_tensor_2

    # Return the results
    return combined_df, provenance

def add_source_identifiers(data_tensor_1, data_tensor_2):
    """
    Add source identifiers to input DataFrames.

    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset.
        data_tensor_2 (pd.DataFrame): Second dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: Updated DataFrames with source identifiers.
    """
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
