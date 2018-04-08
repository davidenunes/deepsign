import tensorflow as tf
import tensorx as tx
import numpy as np
from tensorflow.python.framework import tensor_util

v = np.array([[1, 0, 1], [0, 0, 2], [3, 0, 3], [7, 0, 0]], dtype=np.float32)
ids = np.array([[1, 0], [1, 3]], dtype=np.int64)
print(" dense representation \n ", v)

sp = tx.to_sparse(v)

tf.InteractiveSession()

print(sp.eval())

flat_ids = tf.reshape(ids, [-1])
num_elems = tf.shape(flat_ids)[0]
print("gather_ids: ", flat_ids.eval())

# COUNT COLUMNS
sp_cols = tx.sparse_ones(sp.indices, sp.dense_shape, dtype=tf.int64)
col_count = tf.sparse_reduce_sum(sp_cols, axis=-1)
col_count_cs = tf.cumsum(col_count)
start_coord = col_count_cs - col_count

print("#cols ", col_count.eval())
print("cumsum #cols ", col_count_cs.eval())
print("start coord ", start_coord.eval())

current_coors = None


def expand_row_coors(row_id):
    start = start_coord[row_id]
    num_coors = col_count[row_id]
    row_coors = tf.range(start, start + num_coors, dtype=tf.int64)

    return row_coors


ta_gather = tf.TensorArray(dtype=tf.int32, size=tf.constant(2), infer_shape=False)
loop_i = tf.Variable(0, trainable=False)
# row coordinates computed based on a var value
coors = expand_row_coors(loop_i)

tf.global_variables_initializer().run()

print(coors.eval())

# new_ta = ta.write(loop_i, coors)
# new_i = tf.assign_add(1)

# elems = np.array([1, 1, 1, 1, 1], dtype=np.int32)
# init = tf.constant(10, dtype=tf.int32)
elems = tf.cast(tf.range(0, num_elems), tf.int64)

# init = tf.TensorArray(dtype=tf.int64, size=num_elems, infer_shape=False)
init = tf.constant([], dtype=tf.int64)

# res = tf.foldl(lambda x, elem: x - elem, elems=elems, initializer=init)
res = tf.while_loop(lambda ta, elem: tf.concat([ta, expand_row_coors(elem)], axis=-1), elems=elems, initializer=init,
               shape_invariants=[elems.get_shape(),tf.TensorShape([None])])
print(res)

"""
init = tf.TensorArray(infer_shape=False, size=num_elems, dtype=tf.int64)
row_indices = tf.constant([], dtype=tf.int64)
new_rows = tf.constant([], dtype=tf.int64)
i0 = tf.constant(0)

_, gather_rows, new_rows = tf.while_loop(cond=lambda i, ri, nr: i < num_elems,
                                         body=loop_body,
                                         loop_vars=[i0, row_indices, new_rows],
                                         shape_invariants=[i0.get_shape(), tf.TensorShape([None]),
                                                           tf.TensorShape([None])])

g_c = sp.indices[:, -1]
print("indices: \n ", sp.indices.eval())
new_cols = tf.gather(g_c, gather_rows)
print("gather_rows \n", gather_rows.eval())
print("new_rows \n", new_rows.eval())
print("new cols \n", new_cols.eval())

print(tf.shape(new_rows).eval())
print(tf.shape(new_cols).eval())

new_indices = tf.stack([new_rows, new_cols], axis=-1)
new_values = tf.gather(sp.values, gather_rows)
new_shape = tf.concat([tf.expand_dims(tf.cast(num_elems, tf.int64), -1), sp.dense_shape[1:]], axis=-1)

print("new indices \n", new_indices.eval())
print("new values \n", new_values.eval())
print("new shape \n ", new_shape.eval())

g_sp = tf.SparseTensor(new_indices, new_values, new_shape)

print(g_sp.eval())
"""
