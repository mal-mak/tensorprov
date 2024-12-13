import pandas as pd
import torch
import time

def append_dataframes(data_tensor_1, data_tensor_2):
    """
    Combine two datasets by appending rows.
    
    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset.
        data_tensor_2 (pd.DataFrame): Second dataset.

    Returns:
        combined_df (pd.DataFrame): Combined DataFrame.
        elapsed_time (float): Time taken for the append operation.
    """
    if not isinstance(data_tensor_1, pd.DataFrame):
        raise TypeError("data_tensor_1 must be a Pandas DataFrame.")
    if not isinstance(data_tensor_2, pd.DataFrame):
        raise TypeError("data_tensor_2 must be a Pandas DataFrame.")
    
    if data_tensor_1.empty:
        raise ValueError("data_tensor_1 is empty. Provide a non-empty DataFrame.")
    if data_tensor_2.empty:
        raise ValueError("data_tensor_2 is empty. Provide a non-empty DataFrame.")
    
    start_time = time.time()
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)
    elapsed_time = time.time() - start_time
    return combined_df, elapsed_time

def provenance_matrix_sparse(data_tensor_1, data_tensor_2, combined_df):
    """
    Generate a simple binary provenance matrix indicating the source of each row:
    - '1' for rows from data_tensor_1
    - '0' for rows from data_tensor_2

    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset.
        data_tensor_2 (pd.DataFrame): Second dataset.
        combined_df (pd.DataFrame): Combined dataset after appending.

    Returns:
        provenance (torch.Tensor): Simple binary provenance matrix.
    """
    if combined_df.empty:
        raise ValueError("The combined DataFrame is empty. Something went wrong in the append operation.")
    
    size_1 = len(data_tensor_1)
    size_2 = len(data_tensor_2)
    
    # Create a binary matrix where:
    # - '1' for rows from data_tensor_1
    # - '0' for rows from data_tensor_2
    provenance = torch.zeros((len(combined_df), 2), dtype=torch.int32)
    
    # Set the first column to 1 for rows originating from data_tensor_1
    provenance[:size_1, 0] = 1
    
    # Set the second column to 1 for rows originating from data_tensor_2
    provenance[size_1:, 1] = 1
    
    return provenance

def validate_dataframes(data_tensor_1, data_tensor_2):
    """
    Validate input DataFrames to ensure they are non-empty Pandas DataFrames.

    Parameters:
        data_tensor_1 (pd.DataFrame): First dataset.
        data_tensor_2 (pd.DataFrame): Second dataset.
    """
    if not isinstance(data_tensor_1, pd.DataFrame):
        raise TypeError("data_tensor_1 must be a Pandas DataFrame.")
    if not isinstance(data_tensor_2, pd.DataFrame):
        raise TypeError("data_tensor_2 must be a Pandas DataFrame.")
    if data_tensor_1.empty:
        raise ValueError("data_tensor_1 is empty. Provide a non-empty DataFrame.")
    if data_tensor_2.empty:
        raise ValueError("data_tensor_2 is empty. Provide a non-empty DataFrame.")

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
        float: Time elapsed for the append operation.
    """
    start_time = time.time()
    combined_df = pd.concat([data_tensor_1, data_tensor_2], ignore_index=True)
    elapsed_time = time.time() - start_time
    return combined_df, elapsed_time
