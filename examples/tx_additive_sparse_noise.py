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

labels = tx.Input(1, batch_size=None)
feed = {labels.placeholder: [[1], [2]]}

vocab_size = 100
k = 10
s = 4

generator = Generator(k, s)
ris = [generator.generate() for _ in range(vocab_size)]
ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)

sp = ri_tensor.gather(labels.tensor).to_sparse_tensor()
sp = tf.sparse_tensor_to_dense(sp)

print(sp.eval(feed))

noise = tx.sparse_random_mask(tf.constant(k),
                              tf.shape(labels.tensor)[0],
                              density=0.2,
                              mask_values=[-1, 1],
                              symmetrical=True)
print(noise.eval(feed))
sp2 = tf.sparse_add(sp, noise, thresh=0)

print(sp2.eval(feed))
