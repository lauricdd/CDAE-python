#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17
@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
from utils import data_manager as dp


class Compute_Similarity_Python:

    def __init__(self, dataMatrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine", row_weights=None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions
        """

        super(Compute_Similarity_Python, self).__init__()

        self.shrink = shrink
        self.normalize = normalize

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights),
                                                                                         dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T

    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')

        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        blockSize = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize

    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        blockSize = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize

    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        blockSize = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += blockSize

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()

        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = np.power(sumOfSquared, 2 * self.asymmetric_alpha)

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities => using dot product
                this_block_weights = self.dataMatrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[
                                          columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights) * self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)

            # Add previous block size
            processedItems += this_block_size

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            start_col_block += block_size

        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)

        return W_sparse


def similarityMatrixTopK(item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise
    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

        for item_idx in range(nitems):
            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx + 1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            non_zero_data = column_data != 0

            idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[non_zero_data][top_k_idx])
            rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


# Transforms matrix into a specific format
def check_matrix(X, format='csc', dtype=np.float32):
    """
        This function takes a matrix as input and transforms it into the specified format.
        The matrix in input can be either sparse or ndarray.
        If the matrix in input has already the desired format, it is returned as-is
        the dtype parameter is always applied and the default is np.float32
        :param X:
        :param format:
        :param dtype:
        :return:
    """

    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)  # Compressed Sparse Column format
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)  # Compressed Sparse Row format
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)



from enum import Enum

class SimilarityFunction(Enum):
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"


class Compute_Similarity:


    def __init__(self, dataMatrix, use_implementation = "density", similarity = None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """

        assert np.all(np.isfinite(dataMatrix.data)), \
            "Compute_Similarity: Data matrix contains {} non finite values".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

        else:

            assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
                "Compute_Similarity: data has only 1 feature (shape: {}) with dense values," \
                " vector and set based similarities are not defined on 1-dimensional dense data," \
                " use Euclidean similarity instead.".format(dataMatrix.shape)

            if similarity is not None:
                args["similarity"] = similarity


            if use_implementation == "density":

                if isinstance(dataMatrix, np.ndarray):
                    self.dense = True

                elif isinstance(dataMatrix, sps.spmatrix):
                    shape = dataMatrix.shape

                    num_cells = shape[0]*shape[1]

                    sparsity = dataMatrix.nnz/num_cells

                    self.dense = sparsity > 0.5

                else:
                    print("Compute_Similarity: matrix type not recognized, calling default...")
                    use_implementation = "python"

                if self.dense:
                    print("Compute_Similarity: detected dense matrix")
                    use_implementation = "python"
                else:
                    use_implementation = "cython"


            if use_implementation == "cython":

                try:
                    from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
                    self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)

                except ImportError:
                    print("Unable to load Cython Compute_Similarity, reverting to Python")
                    self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)


            elif use_implementation == "python":
                self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

            else:

                raise  ValueError("Compute_Similarity: value for argument 'use_implementation' not recognized")


    def compute_similarity(self,  **args):

        return self.compute_similarity_object.compute_similarity(**args)




class Compute_Similarity_Euclidean:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize=False, normalize_avg_row=False,
                 similarity_from_distance_mode ="lin", row_weights = None, **args):
        """
        Computes the euclidean similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param normalize
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param similarity_from_distance_mode:       "exp"   euclidean_similarity = 1/(e ^ euclidean_distance)
                                                    "lin"        euclidean_similarity = 1/(1 + euclidean_distance)
                                                    "log"        euclidean_similarity = 1/(1 + euclidean_distance)
        :param args:                accepts other parameters not needed by the current object
        """

        super(Compute_Similarity_Euclidean, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.normalize_avg_row = normalize_avg_row

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.similarity_is_exp = False
        self.similarity_is_lin = False
        self.similarity_is_log = False

        if similarity_from_distance_mode == "exp":
            self.similarity_is_exp = True
        elif similarity_from_distance_mode == "lin":
            self.similarity_is_lin = True
        elif similarity_from_distance_mode == "log":
            self.similarity_is_log = True
        else:
            raise ValueError("Compute_Similarity_Euclidean: value for parameter 'mode' not recognized."
                             " Allowed values are: 'exp', 'lin', 'log'."
                             " Passed value was '{}'".format(similarity_from_distance_mode))



        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Compute_Similarity_Euclidean: provided row_weights and dataMatrix have different number of rows."
                                 "row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T