import tensorflow as tf
from deepsign.rp.ri import RandomIndex
from deepsign.rp.tf_utils import RandomIndexTensor
from tensorflow.python.framework import tensor_util
import tensorx as tx

tf.InteractiveSession()

sp = tf.SparseTensor(indices=[[0, 0]], values=[1], dense_shape=[1, 10])
sp = tf.sparse_reorder(sp)

r = RandomIndex(10, [0, 5], [1, 7])
r2 = RandomIndex(10, [0, 3], [2, 7])

pos, neg = r.sorted_indices_values()

print(pos)
print(neg)

rit = RandomIndexTensor.from_ri_list([r, r2], 10, 4)
rit = rit.gather(0)
print(rit.to_sparse_tensor().eval())

print(rit.to_sparse_tensor().get_shape())

a = 1
b = tf.convert_to_tensor([[1], [2]])

batch = tf.shape(b)[0]
print(batch.eval())

s = tf.stack([batch, tf.convert_to_tensor(a)])
# CAST IS THE PROBLEM, it seems that TF cannot compute a constant value from a casted tensor? why the fuck?
# it can't recover the constant value because of the cast
s = tf.cast(s, tf.int64)
sp = tx.empty_sparse_tensor(s)
print(s.get_shape())

print(tensor_util.constant_value_as_shape(s))
print("compute shape")
print(tf.shape(sp).eval())

# let's try to cast before stacking
a = tf.convert_to_tensor(a, dtype=tf.int64)
b = tf.convert_to_tensor([[1], [2]], dtype=tf.int64)
print(tensor_util.constant_value(a))
print(tensor_util.constant_value(tf.shape(b, out_type=tf.int64)[0]))

s = tf.stack([tf.shape(b, out_type=tf.int64)[0], a])
print(s.get_shape())
print(tensor_util.constant_value_as_shape(s))
