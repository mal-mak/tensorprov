import pandas as pd
import torch
import time


def join(df1, df2, on, how="inner"):
    """
    Effectue une jointure entre deux DataFrames.
    :param df1: Premier DataFrame.
    :param df2: Second DataFrame.
    :param on: Colonne(s) de jointure.
    :param how: Type de jointure (inner, left, right, outer).
    :return: DataFrame résultant.
    """
    result = pd.merge(df1, df2, on=on, how=how)
    return result


def create_provenance_tensor(df1, df2, result_df, on):
    """
    Crée un tenseur de provenance binaire pour capturer l'origine des lignes après la jointure.
    Utilise PyTorch pour un traitement efficace.

    :param df1: Premier DataFrame.
    :param df2: Second DataFrame.
    :param result_df: DataFrame résultant de la jointure.
    :param on: Colonne(s) de jointure.
    :return: Un tenseur binaire 3D (résultat x df1 x df2).
    """
    rows_result = len(result_df)
    rows_df1 = len(df1)
    rows_df2 = len(df2)

    # Convert columns in 'on' to strings (or another uniform type)
    df1[on] = df1[on].astype(str)
    df2[on] = df2[on].astype(str)
    result_df[on] = result_df[on].astype(str)

    # Initialisation du tenseur 3D
    tensor = torch.zeros((rows_result, rows_df1, rows_df2), dtype=torch.int)

    # Remplissage du tenseur
    for i, row in result_df.iterrows():
        # Convert the current row's key to a tuple
        current_key = tuple(row[on].values)

        # Trouver les indices correspondants dans df1 et df2
        match_df1 = df1[on].apply(tuple, axis=1) == current_key
        match_df2 = df2[on].apply(tuple, axis=1) == current_key

        for j, match1 in enumerate(match_df1):
            for k, match2 in enumerate(match_df2):
                if match1 and match2:
                    tensor[i, j, k] = 1

    return tensor


def create_provenance_tensor_hash(df1, df2, result_df, on):
    """
    Creates a binary provenance tensor using hashed keys to capture the origin of rows after a join.
    Utilizes PyTorch for efficient tensor operations.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :param result_df: Resulting DataFrame after the join.
    :param on: Column(s) used for the join.
    :return: A 3D binary tensor (result x df1 x df2) as a PyTorch tensor.
    """
    rows_result = len(result_df)
    rows_df1 = len(df1)
    rows_df2 = len(df2)

    # Ensure the columns used in "on" are treated as strings for consistent hashing
    df1[on] = df1[on].astype(str)
    df2[on] = df2[on].astype(str)
    result_df[on] = result_df[on].astype(str)

    # Generate hash keys for the join columns
    df1_hashes = pd.util.hash_pandas_object(df1[on].apply(tuple, axis=1), index=False)
    df2_hashes = pd.util.hash_pandas_object(df2[on].apply(tuple, axis=1), index=False)
    result_hashes = pd.util.hash_pandas_object(
        result_df[on].apply(tuple, axis=1), index=False
    )

    # Convert hashes to PyTorch tensors
    df1_hashes = torch.tensor(df1_hashes.values, dtype=torch.int64)
    df2_hashes = torch.tensor(df2_hashes.values, dtype=torch.int64)
    result_hashes = torch.tensor(result_hashes.values, dtype=torch.int64)

    # Initialize the provenance tensor
    provenance_tensor = torch.zeros(
        (rows_result, rows_df1, rows_df2), dtype=torch.int32
    )

    # Populate the provenance tensor
    for i, result_hash in enumerate(result_hashes):
        # Find matching indices for df1 and df2
        match_df1 = (df1_hashes == result_hash).nonzero(as_tuple=True)[0]
        match_df2 = (df2_hashes == result_hash).nonzero(as_tuple=True)[0]

        # Update provenance tensor
        for j in match_df1:
            for k in match_df2:
                provenance_tensor[i, j, k] = 1

    return provenance_tensor


def compare_methods(df1, df2, result_df, on):
    """
    Compare the execution times of the two methods for creating a provenance tensor:
    - Row-wise comparison method.
    - Hash-based method.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :param result_df: Resulting DataFrame after the join.
    :param on: Columns used for the join.
    """

    # Measure runtime of row-wise comparison method
    start_row_wise = time.time()
    provenance_tensor = create_provenance_tensor(df1, df2, result_df, on)
    time_row_wise = time.time() - start_row_wise

    # Print result and runtime for the row-wise method
    print("Row-wise Method Result:")
    print(provenance_tensor)
    print(f"Row-wise Method Time: {time_row_wise} seconds")

    # Measure runtime of hash-based method
    start_hash_based = time.time()
    provenance_tensor_hash = create_provenance_tensor_hash(df1, df2, result_df, on)
    time_hash_based = time.time() - start_hash_based

    # Print result and runtime for the hash-based method
    print("\nHash-based Method Result:")
    print(provenance_tensor_hash)
    print(f"Hash-based Method Time: {time_hash_based} seconds")
