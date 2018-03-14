import tensorflow as tf
import tensorx as tx
import numpy as np
from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul
import time

ss = tf.InteractiveSession()

v = np.random.uniform(-1, 1, [3, 6])
v = tf.constant(v, dtype=tf.float32)
print("dense tensor \n ", v.eval())

# create a dummy sparse tensor
sp_v = tx.to_sparse(v)

print(tf.shape(sp_v).eval())

# suppose se are reshaping from a concatenated tensor that represents
# a sequence to a batch of sequences of feature vectors


# print(tf.shape(sp_v_reshaped).eval())

# suppose now that we want to use broadcasting with the sparse_tensor
weights = tf.constant([[1., 0.5], [1., -1.], [1., -1.]], dtype=tf.float32)
weights_r = tx.repeat(weights, 3)

print(weights_r.eval())
# weights = tf.expand_dims(weights, -1)

out = tx.sparse_multiply(sp_v, weights_r)
dense_out = tf.sparse_tensor_to_dense(out)

t0 = time.time()
print(dense_out.eval())
t1 = time.time()
print("time in secs. ", t1 - t0)

sp_vr = tf.sparse_reshape(sp_v, [-1, 2, 3])
out_tf = tx.sparse_multiply_dense(sp_vr, tf.expand_dims(weights, -1))
out_tf = tf.reshape(out_tf, tf.shape(sp_v))
t0 = time.time()
print(out_tf.eval())
t1 = time.time()

print(tf.shape(out_tf).eval())
print("time in secs. ", t1 - t0)
