import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple


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
    Creates a binary provenance tensor to capture the origin of rows after a join.
    """
    rows_result = len(result_df)
    rows_df1 = len(df1)
    rows_df2 = len(df2)

    # Convert to string type for consistent comparison
    df1[on] = df1[on].astype(str)
    df2[on] = df2[on].astype(str)
    result_df[on] = result_df[on].astype(str)

    # Initialize the tensor
    tensor = torch.zeros((rows_result, rows_df1, rows_df2), dtype=torch.int)

    # Populate the tensor
    for i, result_row in enumerate(result_df[on]):
        # Find matching indices in df1 and df2
        match_df1 = df1[on] == result_row
        match_df2 = df2[on] == result_row

        # Update tensor for matching rows
        for j in np.where(match_df1)[0]:
            for k in np.where(match_df2)[0]:
                tensor[i, j, k] = 1

    return tensor

def create_provenance_tensor_hash(df1, df2, result_df, on):
    """
    Creates a binary provenance tensor using hashed keys.
    """
    rows_result = len(result_df)
    rows_df1 = len(df1)
    rows_df2 = len(df2)

    # Convert to string type for consistent hashing
    df1[on] = df1[on].astype(str)
    df2[on] = df2[on].astype(str)
    result_df[on] = result_df[on].astype(str)

    # Generate hash values and convert to int64
    df1_hashes = pd.util.hash_pandas_object(df1[on], index=False).astype(np.int64)
    df2_hashes = pd.util.hash_pandas_object(df2[on], index=False).astype(np.int64)
    result_hashes = pd.util.hash_pandas_object(result_df[on], index=False).astype(np.int64)

    # Convert to PyTorch tensors
    df1_hashes = torch.tensor(df1_hashes.values, dtype=torch.int64)
    df2_hashes = torch.tensor(df2_hashes.values, dtype=torch.int64)
    result_hashes = torch.tensor(result_hashes.values, dtype=torch.int64)

    # Initialize the tensor
    provenance_tensor = torch.zeros((rows_result, rows_df1, rows_df2), dtype=torch.int32)

    # Populate the tensor
    for i, result_hash in enumerate(result_hashes):
        match_df1 = (df1_hashes == result_hash).nonzero(as_tuple=True)[0]
        match_df2 = (df2_hashes == result_hash).nonzero(as_tuple=True)[0]

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


def generate_test_data(sizes: List[int]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate test DataFrames of different sizes.
    """
    test_cases = []
    for size in sizes:
        # Generate random IDs with some overlapping values
        ids1 = np.random.randint(1, size//2 + 1, size=size)
        ids2 = np.random.randint(1, size//2 + 1, size=size)
        
        # Create first DataFrame
        df1 = pd.DataFrame({
            'id': ids1,
            'value1': np.random.rand(size)
        })
        
        # Create second DataFrame
        df2 = pd.DataFrame({
            'id': ids2,
            'value2': np.random.rand(size)
        })
        
        test_cases.append((df1, df2))
    
    return test_cases

def run_performance_tests(test_cases: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> List[dict]:
    """
    Run performance tests on different DataFrame sizes.
    """
    results = []
    
    for df1, df2 in test_cases:
        # Perform join first
        result_df = pd.merge(df1, df2, on='id', how='inner')
        
        # Measure row-wise method
        start_time = time.time()
        create_provenance_tensor(df1, df2, result_df, 'id')
        row_wise_time = time.time() - start_time
        
        # Measure hash-based method
        start_time = time.time()
        create_provenance_tensor_hash(df1, df2, result_df, 'id')
        hash_based_time = time.time() - start_time
        
        results.append({
            'size': len(df1),
            'row_wise_time': row_wise_time,
            'hash_based_time': hash_based_time,
            'result_size': len(result_df)
        })
    
    return results

def plot_performance_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Create performance comparison plots.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Execution Time Comparison
    plt.subplot(2, 2, 1)
    plt.plot(results_df['size'], results_df['row_wise_time'], 'o-', label='Row-wise Method')
    plt.plot(results_df['size'], results_df['hash_based_time'], 'o-', label='Hash-based Method')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison- Execution Time')
    plt.legend()
    plt.grid(True)
    

    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Define test sizes (smaller for initial testing)
    sizes = [10, 50, 100, 200, 50,1000]
    
    print("Generating test data...")
    test_cases = generate_test_data(sizes)
    
    print("Running performance tests...")
    results = run_performance_tests(test_cases)
    
    results_df = pd.DataFrame(results)
    
    print("\nTest Results Summary:")
    print(results_df.to_string(index=False))
    
    print("\nGenerating plots...")
    plot_performance_comparison(results_df, 'performance_report-Join.png')
    
    return results_df

if __name__ == "__main__":
    results_df = main()
