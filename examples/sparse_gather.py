import tensorflow as tf
import tensorx as tx
import numpy as np


def sparse_slice(indices, values, needed_row_ids):
    needed_row_ids = tf.reshape(needed_row_ids, [1, -1])
    num_rows = tf.shape(indices)[0]
    partitions = tf.cast(tf.reduce_any(tf.equal(tf.reshape(indices[:, 0], [-1, 1]), needed_row_ids), 1), tf.int32)
    rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
    slice_indices = tf.gather(indices, rows_to_gather)
    slice_values = tf.gather(values, rows_to_gather)
    return slice_indices, slice_values


v = np.array([[1, 0, 1], [0, 0, 2], [3, 0, 3]], dtype=np.float32)
sp = tx.to_sparse(v)

tf.InteractiveSession()

print(sp.eval())
print(v)
indices = np.array([[0, 1], [0, 0], [1, 2]], dtype=np.int32)
indices = tf.cast(indices, tf.int64)

row_i_raw, col_j = tf.split(sp.indices, 2, axis=-1)
row_i = tf.reshape(row_i_raw, shape=[-1])
col_j = tf.reshape(col_j, shape=[-1])

print("row indices \n", row_i_raw.eval())
print("row indices \n", row_i.eval())

print("indices to gather \n", tf.reshape(indices, [-1, 1]).eval())
row_filter = tf.where(tf.equal(row_i, tf.reshape(indices, [-1, 1])))
new_rows, row_indices = tf.split(row_filter, 2, -1)



# row_indices = tf.reshape(row_indices,[-1])
print("row indices \n", row_indices.eval())

print("new rows \n", new_rows.eval())
# num_rows = tf.reduce_max(new_rows) + 1
num_rows = tf.shape(tf.reshape(indices, [-1]))[-1]
print("num rows \n", num_rows.eval())

column_ids = tf.gather(col_j, row_indices)
print(column_ids.eval())
values = tf.gather_nd(sp.values, row_indices)
# values = tf.reshape(values, [-1])
print("columns \n", column_ids.eval())
print("values \n", values.eval())

row_col = tf.concat([new_rows, column_ids], axis=-1)
print("new indices \n", row_col.eval())

# stack must be 0 to concat into rank 1 tensors
dense_shape = tf.stack([tf.cast(num_rows, tf.int64), sp.dense_shape[-1]])

print("i ", row_col)
print("v ", values)
print("s ", dense_shape)

gather_sp = tf.SparseTensor(indices=row_col, values=values, dense_shape=dense_shape)

dense = tf.sparse_tensor_to_dense(gather_sp)
print(dense.eval())
print(gather_sp.eval())
# dense = tf.sparse_to_dense(row_col, dense_shape, values)
# print(dense.eval())


gather_sp_tx = tx.gather_sparse(sp, indices)

print(tf.sparse_tensor_to_dense(gather_sp_tx).eval())
