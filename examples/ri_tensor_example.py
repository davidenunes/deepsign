import tensorflow as tf
import tensorx as tx
import numpy as np
from deepsign.models.nrp import RandomIndexTensor

indices = np.array([[1, 0], [2, 3], [4, 5], [6, 7], [8, 9]])
signs = np.repeat([[1, -1]], 5, axis=0)

k = 10
s = 2

rit = RandomIndexTensor(indices, signs, k, s)

tf.InteractiveSession()

print(rit.values.eval())
print(rit.indices.eval())

print(rit.to_sparse_tensor().eval())
print(rit.gather([0, 1, 1, 0, 2]).to_sparse_tensor().eval())
