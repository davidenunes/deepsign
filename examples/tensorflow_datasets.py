import tensorflow as tf
import os
import h5py
from deepsign.data.views import chunk_it
import marisa_trie
import tensorx as tx
import numpy as np

home = os.getenv("HOME")
corpus_file = home + "/data/datasets/ptb/ptb.hdf5"

batch_size = 50
num_workers = 8

corpus_hdf5 = h5py.File(corpus_file, mode='r')
data = corpus_hdf5["validation"]

vocab = marisa_trie.Trie(corpus_hdf5["vocabulary"])
vocab_size = len(vocab)

print("reading from hdf5 dataset")
print(data[:2])


# ngram generator
def get_ngrams():
    for ngram in chunk_it(data, chunk_size=batch_size * 100):
        yield ngram


ds = tf.data.Dataset.from_generator(get_ngrams, tf.int64)
# shards the dataset into unique num_workers unique shards, if I nrp this in MPI this will become useful
# ds = ds.shard(num_workers,0)
ds = ds.repeat(2)
# ds = ds.shuffle(10)
ds = ds.batch(2)
value = ds.make_one_shot_iterator().get_next()
ctx_tensor, w = tf.split(value, [3, 1], axis=-1)
w = tf.reshape(w, shape=[-1])

one_hot = tf.one_hot(w, vocab_size)
where_one = tf.where(tf.equal(one_hot, 1))

"""
Have to be careful with multiple calls to nrp
if we call nrp multiple times it takes values from the dataset iterator
"""
with tf.Session() as sess:
    ctx, w_i, one_hot, where_one = sess.run([ctx_tensor, w, one_hot, where_one])
    # w = sess.nrp(value[:-1])

    print("ctx")
    print(ctx)

    print("w_i")
    print(w_i)

    print("one hot")

    print(np.nonzero(one_hot))

    print("indices")
    print(where_one)
