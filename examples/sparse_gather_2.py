import tensorflow as tf
import tensorx as tx
import numpy as np

v = np.array([[1, 0, 1], [0, 0, 2], [3, 0, 3]], dtype=np.float32)
ids = np.array([[0, 1], [0, 0], [1, 2]], dtype=np.int64)
print(" dense representation \n ", v)

sp = tx.to_sparse(v)

tf.InteractiveSession()



print(sp.eval())

flat_ids = tf.reshape(ids, [-1])
print(flat_ids.eval())
r = tf.gather(sp,indices=[0],axis=0)
print(r.eval())

# print(tf.sparse_tensor_to_dense(sp_t).eval())
