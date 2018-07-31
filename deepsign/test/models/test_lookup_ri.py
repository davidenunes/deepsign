import tensorflow as tf
import tensorx as tx
from deepsign.models.nrp import RandomIndexTensor
from deepsign.rp.ri import Generator, RandomIndex
import numpy as np

sess = tf.InteractiveSession()

vocab_size = 8
k = 6
s = 2
emebd = 3

generator = Generator(k, s)
ris = [generator.generate() for _ in range(vocab_size)]
ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
ri_input = ri_tensor.gather([[0, 1, 0],[1,2,0]])

sp = ri_input.to_sparse_tensor()
sp = tx.TensorLayer(sp, k)
print(sp.tensor.eval())

embed = tx.Lookup(sp, seq_size=3, lookup_shape=[k, 3])


tf.global_variables_initializer().run()

print(np.shape(embed.tensor.eval()))