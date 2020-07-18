#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/09/2018
@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np

class IncrementalSparseMatrix_ListBased(object):

    def __init__(self, auto_create_col_mapper=False, auto_create_row_mapper=False, n_rows=None, n_cols=None):

        super(IncrementalSparseMatrix_ListBased, self).__init__()

        self._row_list = []
        self._col_list = []
        self._data_list = []

        self._n_rows = n_rows
        self._n_cols = n_cols
        self._auto_create_column_mapper = auto_create_col_mapper
        self._auto_create_row_mapper = auto_create_row_mapper

        if self._auto_create_column_mapper:
            self._column_original_ID_to_index = {}

        if self._auto_create_row_mapper:
            self._row_original_ID_to_index = {}

    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add), \
            "IncrementalSparseMatrix: element lists must have different length"

        col_list_index = [self._get_column_index(column_id) for column_id in col_list_to_add]
        row_list_index = [self._get_row_index(row_id) for row_id in row_list_to_add]

        self._row_list.extend(row_list_index)
        self._col_list.extend(col_list_index)
        self._data_list.extend(data_list_to_add)

    def add_single_row(self, row_id, col_list, data=1.0):

        n_elements = len(col_list)

        col_list_index = [self._get_column_index(column_id) for column_id in col_list]
        row_index = self._get_row_index(row_id)

        self._row_list.extend([row_index] * n_elements)
        self._col_list.extend(col_list_index)
        self._data_list.extend([data] * n_elements)

    def get_column_token_to_id_mapper(self):

        if self._auto_create_column_mapper:
            return self._column_original_ID_to_index.copy()

        dummy_column_original_ID_to_index = {}

        for col in range(self._n_cols):
            dummy_column_original_ID_to_index[col] = col

        return dummy_column_original_ID_to_index

    def get_row_token_to_id_mapper(self):

        if self._auto_create_row_mapper:
            return self._row_original_ID_to_index.copy()

        dummy_row_original_ID_to_index = {}

        for row in range(self._n_rows):
            dummy_row_original_ID_to_index[row] = row

        return dummy_row_original_ID_to_index

    def _get_column_index(self, column_id):

        if not self._auto_create_column_mapper:
            column_index = column_id

        else:

            if column_id in self._column_original_ID_to_index:
                column_index = self._column_original_ID_to_index[column_id]

            else:
                column_index = len(self._column_original_ID_to_index)
                self._column_original_ID_to_index[column_id] = column_index

        return column_index

    def _get_row_index(self, row_id):

        if not self._auto_create_row_mapper:
            row_index = row_id

        else:

            if row_id in self._row_original_ID_to_index:
                row_index = self._row_original_ID_to_index[row_id]

            else:
                row_index = len(self._row_original_ID_to_index)
                self._row_original_ID_to_index[row_id] = row_index

        return row_index

    def get_nnz(self):
        return len(self._row_list)

    def get_SparseMatrix(self):

        if self._n_rows is None:
            self._n_rows = max(self._row_list) + 1

        if self._n_cols is None:
            self._n_cols = max(self._col_list) + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_list, (self._row_list, self._col_list)), shape=shape)
        sparseMatrix.eliminate_zeros()

        return sparseMatrix


class IncrementalSparseMatrix(IncrementalSparseMatrix_ListBased):

    def __init__(self, auto_create_col_mapper=False, auto_create_row_mapper=False, n_rows=None, n_cols=None,
                 dtype=np.float64):

        super(IncrementalSparseMatrix, self).__init__(auto_create_col_mapper=auto_create_col_mapper,
                                                      auto_create_row_mapper=auto_create_row_mapper,
                                                      n_rows=n_rows,
                                                      n_cols=n_cols)

        self._dataBlock = 10000000
        self._next_cell_pointer = 0

        self._dtype_data = dtype
        self._dtype_coordinates = np.uint32
        self._max_value_of_coordinate_dtype = np.iinfo(self._dtype_coordinates).max

        self._row_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._col_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._data_array = np.zeros(self._dataBlock, dtype=self._dtype_data)

    def get_nnz(self):
        return self._next_cell_pointer

    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add), \
            "IncrementalSparseMatrix: element lists must have the same length"

        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate(
                    (self._row_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate(
                    (self._col_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._dataBlock, dtype=self._dtype_data)))

            row_index = self._get_row_index(row_list_to_add[data_point_index])
            col_index = self._get_column_index(col_list_to_add[data_point_index])

            self._row_array[self._next_cell_pointer] = row_index
            self._col_array[self._next_cell_pointer] = col_index
            self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

            self._next_cell_pointer += 1

    def add_single_row(self, row_index, col_list, data=1.0):

        n_elements = len(col_list)

        self.add_data_lists([row_index] * n_elements,
                            col_list,
                            [data] * n_elements)

    def get_SparseMatrix(self):
        if self._n_rows is None:
            self._n_rows = self._row_array.max() + 1

        if self._n_cols is None:
            self._n_cols = self._col_array.max() + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_array[:self._next_cell_pointer],
                                       (self._row_array[:self._next_cell_pointer],
                                        self._col_array[:self._next_cell_pointer])),
                                      shape=shape,
                                      dtype=self._dtype_data)

        sparseMatrix.eliminate_zeros()

        return sparseMatrix


def assert_disjoint_matrices(URM_list):
    """
    Checks whether the URM in the list have an empty intersection, therefore there is no data point contained in more than one
    URM at a time
    :param URM_list:
    :return:
    """

    URM_implicit_global = None

    cumulative_nnz = 0

    for URM in URM_list:

        cumulative_nnz += URM.nnz
        URM_implicit = URM.copy()
        URM_implicit.data = np.ones_like(URM_implicit.data)

        if URM_implicit_global is None:
            URM_implicit_global = URM_implicit

        else:
            URM_implicit_global += URM_implicit

    assert cumulative_nnz == URM_implicit_global.nnz, \
        "assert_disjoint_matrices: URM in list are not disjoint, {} data points are in more than one URM".format(
            cumulative_nnz - URM_implicit_global.nnz)

    return True

