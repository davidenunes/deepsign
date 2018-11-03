import numpy as np
from tensorflow import SparseTensorValue
import tensorflow as tf
from tensorx.utils import to_tensor_cast
import tensorx as tx
from tensorflow.python.framework import tensor_util


def ris_to_sp_tensor_value(ri_seq, dim, all_positive=False):
    """ RandomIndex to SparseTensorValue

    Args:
        dim: dimension for the sparse representation should all be the same in ri_seq
        ri_seq: a list of RandomIndex instances
        all_positive: converts the random indices to sparse random vectors with positive entries only (to be used
        as a counter example to random indexing)
    Returns:
        SparseTensorValue
    """
    ri_seq = list(ri_seq)
    sp_indices = []
    sp_values = []
    for i, ri in enumerate(ri_seq):
        indices, values = ri.sorted_indices_values()
        sp_indices.extend(zip([i] * len(indices), indices))
        if all_positive:
            values = list(map(abs, values))

        sp_values.extend(values)

    sp_indices = np.array(sp_indices, np.int64)
    sp_values = np.array(sp_values, np.float32)
    dense_shape = np.array([len(ri_seq), dim], np.int64)

    return SparseTensorValue(sp_indices, sp_values, dense_shape)


class RandomIndexTensor:
    def __init__(self, indices, values, k, s, dtype=tf.float32):
        self.indices = to_tensor_cast(indices, tf.int64)
        self.values = to_tensor_cast(values)
        self.k = k
        self.s = s

    @staticmethod
    def from_ri_list(ri_list, k, s, dtype=tf.float32):
        iv = [ri.sorted_indices_values() for ri in ri_list]
        indices, values = zip(*iv)
        return RandomIndexTensor(indices, values, k, s, dtype)

    def to_sparse_tensor(self, reorder=False):
        with tf.name_scope("to_sparse"):
            """
            [[0,2],[0,4]] ---> [[0,0],[0,2],[1,0],[1,4]]
            """
            indices = tx.column_indices_to_matrix_indices(self.indices, dtype=tf.int64)
            values = tf.reshape(self.values, [-1])

            # num_rows = self.indices.get_shape().as_list()[0]
            num_rows = tf.shape(self.indices, out_type=tf.int64)[0]
            num_cols = tf.convert_to_tensor(self.k, dtype=tf.int64)

            dense_shape = tf.stack([num_rows, num_cols])

            sp = tf.SparseTensor(indices, values, dense_shape)
            if reorder:
                sp = tf.sparse_reorder(sp)

        return sp

    def gather(self, ids):
        with tf.name_scope("gather"):
            ids = to_tensor_cast(ids, tf.int64)
            ids = tf.reshape(ids, [-1])
            indices = tf.gather(self.indices, ids)
            values = tf.gather(self.values, ids)

        return RandomIndexTensor(indices, values, self.k, self.s)
