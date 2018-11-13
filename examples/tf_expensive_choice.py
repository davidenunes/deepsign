import tensorflow as tf
import numpy as np
import os
import tensorx as tx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ss = tf.InteractiveSession()

# sample without replacement using gumbel max trick

n = 10
b = 20
sample = 10


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


# r = tf.random.uniform([b, n])
# z = sample_gumbel([b,n])
# _, indices = tf.nn.top_k(z,k=sample)
# print(z.eval())


# i = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, b])
# s = tf.random_shuffle(i)
# sample = s[:sample]
# print(s.eval())
# print(i.eval())
# print(s.eval())



samples = tx.choice(10, 4, batch_size=4, dtype=tf.int64)
s = tf.shape(samples, out_type=tf.int64)

indices = tf.range(0, s[0])
indices = tx.repeat_each(indices, s[1])
indices = tf.stack([tf.reshape(indices,[-1]),tf.reshape(samples,[-1])],axis=-1)

sx, ix = ss.run([samples,indices])
print(sx)
print(ix)
