import numpy as np
from tensorflow import SparseTensorValue
from deepsign.rp.ri import RandomIndex


def batch_one_hot(indices, dim, dtype=np.int64):
    """

    Args:
        dtype: the output dtype for the tensor
        indices: a list, an [N] shaped array, or [N,1] shaped array with the ids to be encoded as a batch of one hot
        encoded arrays
        dim: the dimension for the final encoding

    Returns:
        an [N,dim] dimensional array with the batch of one-hot encoding for the given ids

    """
    if len(np.shape(indices)) > 1:
        indices = np.reshape(indices, [-1])

    batch_size = len(indices)
    base = np.arange(batch_size, dtype=dtype)
    base *= dim

    indices += base

    one_hot = np.zeros([batch_size, dim], dtype=dtype)
    one_hot.put(indices, 1)

    return one_hot


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
