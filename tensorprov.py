import filter.horizontal_reduction as hr
import filter.vertical_reduction as vr
import join.join as j
import oversampling.oversampling as o
import union.union as u


class Tensorprov:
    """
    Class for applying various transformations (filtering, joining, oversampling, union) 
    on tensors (DataFrames) and tracking the provenance of these transformations.
    """
    def __init__(self):
        """
        Initializes the Tensorprov class.
        """
        pass

    # Transform methods:
    # Filter: Horizontal Reduction
    def horizontal_reduction_transform(self, input, mask):
        """
        Applies horizontal reduction transformation to the input tensor (DataFrame).
        
        Args:
            input (pd.DataFrame): The input DataFrame to transform.
            mask (list): The mask to apply to the input DataFrame.

        Returns:
            pd.DataFrame: The horizontally reduced DataFrame.
        """
        return hr.transform(input, mask)

    # Filter: Vertical Reduciton
    def vertical_reduction_transform(self, input, columns, retain=False):
        """
        Applies vertical reduction transformation to the input tensor (DataFrame).
        
        Args:
            input (pd.DataFrame): The input DataFrame to transform.
            columns (list): The list of columns to retain or reduce.
            retain (bool): Whether to retain the selected columns (default is False).

        Returns:
            pd.DataFrame: The vertically reduced DataFrame.
        """
        return vr.transform(input, columns, retain)

    # Join
    def join_transform(self, input1, input2, join_on, how="inner"):
        """
        Applies a join transformation between two input tensors (DataFrames).
        
        Args:
            input1 (pd.DataFrame): The first DataFrame to join.
            input2 (pd.DataFrame): The second DataFrame to join.
            join_on (str): The column to join on.
            how (str): The type of join to perform, e.g., 'inner', 'outer' (default is 'inner').

        Returns:
            pd.DataFrame: The resulting joined DataFrame.
        """
        return j.join(input1, input2, join_on, how)

    # Oversampling
    def oversample_transform(self, input, method="horizontal", factor=2):
        """
        Applies oversampling transformation to the input tensor (DataFrame).
        
        Args:
            input (pd.DataFrame): The input DataFrame to transform.
            method (str): The oversampling method, either 'horizontal' or 'vertical' (default is 'horizontal').
            factor (int): The oversampling factor (default is 2).

        Returns:
            pd.DataFrame: The oversampled DataFrame.
        """
        return o.oversample(input, method, factor)

    # ----------------------------

    # Provenance methods:
    # Filter: Horizontal Reduction
    def provenance_horizontal_reduction_index_matching(
        self, input, output, sparse=True
    ):
        """
        Tracks the provenance of a horizontal reduction transformation using index matching.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the transformation.
            output (pd.DataFrame): The resulting DataFrame after the transformation.
            sparse (bool): Whether to return the sparse version of the provenance (default is True).

        Returns:
            torch.Tensor: The provenance tensor for the horizontal reduction (index matching).
        """
        return hr.provenance_index_matching(input, output, sparse)

    def provenance_horizontal_reduction_hashing(self, input, output, sparse=True):
        """
        Tracks the provenance of a horizontal reduction transformation using hashing.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the transformation.
            output (pd.DataFrame): The resulting DataFrame after the transformation.
            sparse (bool): Whether to return the sparse version of the provenance (default is True).

        Returns:
            torch.Tensor: The provenance tensor for the horizontal reduction (hashing).
        """
        return hr.provenance_by_hashing(input, output, sparse)

    # Filter: Vertical Reduction
    def provenance_vertical_reduction_column_matching(self, input, output, sparse=True):
        """
        Tracks the provenance of a vertical reduction transformation using column matching.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the transformation.
            output (pd.DataFrame): The resulting DataFrame after the transformation.
            sparse (bool): Whether to return the sparse version of the provenance (default is True).

        Returns:
            torch.Tensor: The provenance tensor for the vertical reduction (column matching).
        """
        return vr.provenance_column_matching(input, output, sparse)

    def provenance_vertical_reduction_hashing(self, input, output, sparse=True):
        """
        Tracks the provenance of a vertical reduction transformation using hashing.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the transformation.
            output (pd.DataFrame): The resulting DataFrame after the transformation.
            sparse (bool): Whether to return the sparse version of the provenance (default is True).

        Returns:
            torch.Tensor: The provenance tensor for the vertical reduction (hashing).
        """
        return vr.provenance_by_hashing(input, output, sparse)

    # Join
    def provenance_join(self, input1, input2, output, join_on):
        """
        Tracks the provenance of a join transformation.
        
        Args:
            input1 (pd.DataFrame): The first DataFrame before the join.
            input2 (pd.DataFrame): The second DataFrame before the join.
            output (pd.DataFrame): The resulting DataFrame after the join.
            join_on (str): The column on which the join was performed.

        Returns:
            torch.Tensor: The provenance tensor for the join transformation.
        """
        return j.create_provenance_tensor_hash(input1, input2, output, join_on)

    def provenance_join_hashing(self, input1, input2, output, join_on):
        """
        Tracks the provenance of a join transformation using hashing.
        
        Args:
            input1 (pd.DataFrame): The first DataFrame before the join.
            input2 (pd.DataFrame): The second DataFrame before the join.
            output (pd.DataFrame): The resulting DataFrame after the join.
            join_on (str): The column on which the join was performed.

        Returns:
            torch.Tensor: The provenance tensor for the join transformation (hashing).
        """
        return j.create_provenance_tensor_hash(input1, input2, output, join_on)

    # Oversampling
    def provenance_oversample_sparse(self, input, output, method="horizontal"):
        """
        Tracks the provenance of an oversampling transformation using sparse representation.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the oversampling.
            output (pd.DataFrame): The resulting DataFrame after the oversampling.
            method (str): The oversampling method used, either 'horizontal' or 'vertical'.

        Returns:
            torch.Tensor: The sparse provenance tensor for the oversampling transformation.
        """
        return o.determine_provenance_sparse(input, output, method)

    def provenance_oversample_dense(self, input, output, method="horizontal"):
        """
        Tracks the provenance of an oversampling transformation using dense representation.
        
        Args:
            input (pd.DataFrame): The input DataFrame before the oversampling.
            output (pd.DataFrame): The resulting DataFrame after the oversampling.
            method (str): The oversampling method used, either 'horizontal' or 'vertical'.

        Returns:
            torch.Tensor: The dense provenance tensor for the oversampling transformation.
        """
        return o.determine_provenance_dense(input, output, method)

    # Union
    def provenance_union(self, input1, input2):
        """
        Tracks the provenance of a union transformation.
        
        Args:
            input1 (pd.DataFrame): The first DataFrame in the union.
            input2 (pd.DataFrame): The second DataFrame in the union.

        Returns:
            torch.Tensor: The provenance tensor for the union transformation.
        """
        return u.sparse_tensor_prov(input1, input2)

    def provenance_with_df_union(self, input1, input2):
        """
        Tracks the provenance of a union transformation with DataFrame source identifiers.
        
        Args:
            input1 (pd.DataFrame): The first DataFrame in the union.
            input2 (pd.DataFrame): The second DataFrame in the union.

        Returns:
            pd.DataFrame: The union of the input DataFrames with added provenance information.
        """
        data1, data2 = u.add_source_identifiers(input1, input2)
        return u.append_with_provenance(data1, data2)
