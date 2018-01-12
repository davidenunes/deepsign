import numpy as np


def batch_one_hot(indices, dim, dtype=np.int64):
    """

    Args:
        indices: a list, an [N] shaped array, or [N,1] shaped array with the ids to be encoded as a batch of one hot
        encoded arrays
        dim: the dimension for the final encoding

    Returns:
        an [N,dim] dimensional array with the batch of one-hot encoding for the given ids

    """
    if len(np.shape(indices)) > 1:
        indices = np.reshape(indices, [-1])

    batch_size = len(indices)
    base = np.arange(batch_size,dtype=dtype)
    base *= dim

    indices += base

    one_hot = np.zeros([batch_size, dim], dtype=dtype)
    one_hot.put(indices, 1)

    return one_hot
