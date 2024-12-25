import filter.horizontal_reduction as hr
import filter.vertical_reduction as vr
import join.join as j
import oversampling.oversampling as o
import union.union as u


class Tensorprov:
    def __init__(self):
        pass

    # Transform methods:
    # Filter: Horizontal Reduction
    def horizontal_reduction_transform(self, input, mask):
        return hr.transform(input, mask)

    # Filter: Vertical Reduciton
    def vertical_reduction_transform(self, input, columns, retain=False):
        return vr.transform(input, columns, retain)

    # Join
    def join_transform(self, input1, input2, join_on, how="inner"):
        return j.join(input1, input2, join_on, how)

    # Oversampling
    def oversample_transform(self, input, method="horizontal", factor=2):
        return o.oversample(input, method, factor)

    # ----------------------------

    # Provenance methods:
    # Filter: Horizontal Reduction
    def provenance_horizontal_reduction_index_matching(
        self, input, output, sparse=True
    ):
        return hr.provenance_index_matching(input, output, sparse)

    def provenance_horizontal_reduction_hashing(self, input, output, sparse=True):
        return hr.provenance_by_hashing(input, output, sparse)

    # Filter: Vertical Reduction
    def provenance_vertical_reduction_column_matching(self, input, output, sparse=True):
        return vr.provenance_column_matching(input, output, sparse)

    def provenance_vertical_reduction_hashing(self, input, output, sparse=True):
        return vr.provenance_by_hashing(input, output, sparse)

    # Join
    def provenance_join(self, input1, input2, output, join_on):
        return j.create_provenance_tensor_hash(input1, input2, output, join_on)

    def provenance_join_hashing(self, input1, input2, output, join_on):
        return j.create_provenance_tensor_hash(input1, input2, output, join_on)

    # Oversampling
    def provenance_oversample_sparse(self, input, output, method="horizontal"):
        return o.determine_provenance_sparse(input, output, method)

    def provenance_oversample_dense(self, input, output, method="horizontal"):
        return o.determine_provenance_dense(input, output, method)

    # Union
    def provenance_union(self, input1, input2):
        return u.sparse_tensor_prov(input1, input2)

    def provenance_with_df_union(self, input1, input2):
        data1, data2 = u.add_source_identifiers(input1, input2)
        return u.append_with_provenance(data1, data2)
