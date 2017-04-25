import numpy as np
import tensorflow as tf


def indices_to_sparse(indices, shape):
    """
    Converts a list of lists of indexes to a sparse tensor value with the given shape
    
    example: 
    
    idx =[[0,5],[0,2,7],[1]]
    
    we want to transform this into:
    
    SparseTensorValue(indices=[[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]], 
                 values=[0,5,0,2,7,1], 
                 dense_shape=[3,10])
                 
    this can be then fed to a tf.sparse_placeholder
    
    if any index value >= shape[1] it raises an exception
    
    :param indices: list of lists of indexes
    :param shape: the given shape, typically [BATCH_SIZE, MAX_INDEX]
    :return: a sparse tensor with the sparse indexes
    """
    idx = []
    for row, iv in enumerate(indices):
        for i in iv:
            if i >= shape[1]:
                raise Exception("Invalid shape: index value " + i + " >= ", shape[1])
            idx.append([row, i])
    idx = np.array(idx)
    values = np.array(sum(indices, []))

    return tf.SparseTensorValue(indices=idx, values=values, dense_shape=shape)


def values_to_sparse(values, sp_indices, shape):
    """ Converts a list of value vectors to a sparse tensor value, maps each index in 
    the given sp_indices to each value. 

    sp_indices have the form of an array [[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]]
    
    :param values: values to be encapsulated by the sparse tensor value
    :param sp_indices: indices to be mapped to each value
    :param shape: given shape of the sparse tensor value
    :return: a sparse tensor value with each index mapping to the given values
    """
    if len(sp_indices) != len(values):
        raise Exception("Number of indices doesn't match number of values: " + len(sp_indices) + "!=" + len(values))

    return tf.SparseTensorValue(indices=sp_indices, values=values, dense_shape=shape)
