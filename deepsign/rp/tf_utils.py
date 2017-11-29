import numpy as np
import tensorflow as tf


def to_sparse_tensor_value(ri_seq,dim):
    """

    :param dim: dimension for the sparse representation
    should correspond to ri in the seq
    :param ri_seq: a list of RandomIndex instances
    :return:
    """
    sp_indices = []
    sp_values = []
    for i, ri in enumerate(ri_seq):
        indices, values = ri.sorted_indices_values()
        sp_indices.extend(zip([i]*len(indices),indices))
        sp_values.extend(values)

    sp_indices = np.array(sp_indices,np.int64)
    sp_values = np.array(sp_values,np.float32)
    dense_shape = np.array([len(ri_seq),dim],np.int64)

    return tf.SparseTensorValue(sp_indices,sp_values,dense_shape)

