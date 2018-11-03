import tensorflow as tf
import tensorx as tx
from deepsign.models.ri_nce import RandomIndexTensor
from deepsign.rp.index import Generator
from tensorflow.python.ops.nn import embedding_lookup_sparse
import os
import numpy as np
from tensorflow.python.framework import tensor_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.InteractiveSession()

labels = [[0, 1], [2, 3]]

flat_labels = tf.reshape(labels, [-1])

vocab_size = 1000
k = 100
s = 2
embed_size = 4

generator = Generator(k, s)
ris = [generator.generate() for _ in range(vocab_size)]
ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)

sp_values = ri_tensor.gather(flat_labels).to_sparse_tensor()
sp_indices = tx.sparse_indices(sp_values)

print(sp_values.get_shape())
print(tensor_util.constant_value_as_shape(sp_values.dense_shape))
print(tensor_util.constant_value(sp_values.dense_shape))
print(sp_values.dense_shape[-1].eval())
print(tf.shape(sp_values).eval())

lookup = tx.Lookup(tx.TensorLayer(sp_values),
                   seq_size=1,
                   lookup_shape=[k, embed_size])

linear = tx.Linear(tx.TensorLayer(sp_values), n_units=k, shared_weights=lookup.weights)

w = embedding_lookup_sparse(
    params=lookup.weights,
    sp_ids=sp_indices,
    sp_weights=sp_values,
    combiner="sum",
    partition_strategy="mod")

tf.global_variables_initializer().run()

np.testing.assert_array_equal(w.eval(), tx.Flatten(lookup).eval())
np.testing.assert_array_equal(w.eval(), linear.eval())
