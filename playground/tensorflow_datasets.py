import tensorflow as tf
import os
import h5py
from deepsign.data.views import chunk_it

home = os.getenv("HOME")
corpus_file = home + "/data/datasets/ptb/ptb.hdf5"

batch_size = 50
num_workers = 8

corpus_hdf5 = h5py.File(corpus_file, mode='r')
data = corpus_hdf5["validation"]

print("reading from hdf5 dataset")
print(data[:2])


# ngram generator
def get_ngrams():
    for ngram in chunk_it(data, chunk_size=batch_size * 100):
        yield ngram


ds = tf.data.Dataset.from_generator(get_ngrams, tf.int64)
# shards the dataset into unique num_workers unique shards, if I run this in MPI this will become useful
# ds = ds.shard(num_workers,0)
ds = ds.repeat(2)
# ds = ds.shuffle(10)
ds = ds.batch(2)
value = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    ctx, w = sess.run(tf.split(value, [3, 1], axis=-1))
    # w = sess.run(value[:-1])

    print(ctx)
    print(w)
