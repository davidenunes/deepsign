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


# init = tf.Variable(initial_value=[], trainable=False, validate_shape=False, dtype=tf.int64)
# tf.global_variables_initializer().run()

# assign = tf.assign(init, [1], validate_shape=False)
# assign.eval()
# print(init.eval())


# gather_coors = tf.scan(process_rows, flat_ids, initializer=init, shape_invariants=True)


# def loop_body(i, ta):
#    ris = expand_row_coors(flat_ids[i])
#    return i + 1, ta.write(i, ris)


def loop_body_v2(i, prev, n_rows):
    ris = expand_row_coors(flat_ids[i])
    ni = tf.cast(i, tf.int64)
    row_size = col_count[flat_ids[i]]
    new_rows = tf.tile(tf.expand_dims(ni, -1), tf.expand_dims(row_size, -1))
    return i + 1, tf.concat([prev, ris], axis=-1), tf.concat([n_rows, new_rows], axis=-1)


num_elems = tf.shape(flat_ids)[0]
init = tf.TensorArray(infer_shape=False, size=num_elems, dtype=tf.int64)
row_indices = tf.constant([], dtype=tf.int64)
new_rows = tf.constant([], dtype=tf.int64)
i0 = tf.constant(0)

_, gather_rows, new_rows = tf.while_loop(cond=lambda i, ri, nr: i < num_elems,
                                         body=loop_body_v2,
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
