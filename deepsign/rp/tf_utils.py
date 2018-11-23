import numpy as np
import tensorflow as tf
from tensorx.utils import to_tensor_cast
import tensorx as tx


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
            indices = tx.to_matrix_indices(self.indices, dtype=tf.int64)
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


def generate_noise(k_dim, batch_size, ratio=0.01):
    """ generates sparse symmetric noise

    Args:
        k_dim:
        batch_size:
        ratio:

    Returns:

    """
    num_noise = min(int(k_dim * ratio) // 2 * 2, 2)

    i = np.array([np.random.choice(k_dim, size=num_noise, replace=False) for _ in range(batch_size)])
    v = np.concatenate([np.full([batch_size, num_noise // 2], fill_value=1.),
                        np.full([batch_size, num_noise // 2], fill_value=-1.)],
                       axis=-1)

    sorted_i = np.argsort(i, axis=-1)
    rows = np.reshape(np.repeat(np.arange(batch_size), num_noise), [-1, 2])
    i = i[rows, sorted_i]
    v = v[rows, sorted_i]

    rows = rows.flatten()
    i = i.flatten()
    values = v.flatten()
    indices = np.stack([rows, i], axis=-1)


    return tf.SparseTensorValue(indices, values, dense_shape=[batch_size, k_dim])
