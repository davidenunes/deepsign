import numpy as np
import tensorflow as tf
import tensorx as tx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
B = tf.constant([[5, 6], [7, 8], [9, 10]], dtype=tf.float32)
C = tf.SparseTensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], tf.constant([5, 6, 7, 8, 9, 10], tf.float32),
                    [3, 2])

r1 = tf.matmul(A, B, transpose_b=True)

rs = tf.sparse_tensor_dense_matmul(C, A, adjoint_b=True)
rs = tf.transpose(rs)

D = tf.sparse_tensor_to_dense(C)
r2 = tf.sparse_matmul(A, B, transpose_b=True, b_is_sparse=True)

sess = tf.Session()

Cd = tf.sparse_tensor_to_dense(C)
r3 = tf.matmul(A, Cd, b_is_sparse=True, transpose_b=True)

#Ct = tf.sparse_transpose(C)
Ct = C
Ci = tx.sparse_indices(Ct)
r4 = tf.nn.embedding_lookup_sparse(tf.transpose(A),sp_ids=Ci,sp_weights=Ct,combiner="sum")

print(sess.run(r1))
print("=" * 40)
print(sess.run(rs))
print("=" * 40)
print(sess.run(r2))
print("=" * 40)
print(sess.run(r3))
print("=" * 40)
print(sess.run(r4))

