"Oversampling tensors"
# pylint: disable=no-member
import torch

def oversampling(data, method='horizontal', factor=2):
    """
    Perform oversampling (horizontal or vertical) on the input data using PyTorch.

    Parameters:
        data (torch.Tensor): Input dataset (2D tensor).
        method (str): Type of oversampling ('horizontal' or 'vertical').
        factor (int): Multiplication factor for oversampling.

    Returns:
        torch.Tensor: Augmented dataset.
    """
    if method == 'horizontal':
        # Horizontal augmentation: replicate columns
        augmented_data = data.repeat(1, factor)
    elif method == 'vertical':
        # Vertical augmentation: replicate rows
        augmented_data = data.repeat(factor, 1)
    else:
        raise ValueError("Invalid method. Choose 'horizontal' or 'vertical'.")   
    return augmented_data

def tensrov(original_data, augmented_data):
    """
    Generate a sparse binary tensor to capture the provenance of the oversampling operation.

    Parameters:
        original_data (torch.Tensor): Original input dataset.
        augmented_data (torch.Tensor): Output dataset after oversampling.

    Returns:
        torch.sparse.Tensor: Sparse binary tensor capturing provenance.
    """
    original_rows, original_cols = original_data.size()
    augmented_rows, augmented_cols = augmented_data.size()

    if augmented_rows > original_rows:
        # Vertical augmentation
        row_map = torch.arange(augmented_rows) % original_rows
        indices = torch.stack([row_map, torch.arange(augmented_rows)])
    elif augmented_cols > original_cols:
        # Horizontal augmentation
        col_map = torch.arange(augmented_cols) % original_cols
        indices = torch.stack([torch.arange(augmented_cols), col_map])
    else:
        raise ValueError("Augmented data dimensions must differ from the original data dimensions.")

    # Create values for the sparse tensor (all ones for binary provenance tracking)
    values = torch.ones(indices.size(1), dtype=torch.float32)

    # Create the sparse provenance tensor
    provenance_tensor = torch.sparse_coo_tensor(
        indices, values, size=(max(original_rows, original_cols), max(augmented_rows, augmented_cols))
    )
    return provenance_tensor
