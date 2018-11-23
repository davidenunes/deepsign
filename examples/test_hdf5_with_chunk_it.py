import tensorflow as tf
import os
import h5py
import time
from deepsign.data.iterators import chunk_it
from tqdm import tqdm

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

home = os.getenv("HOME")
corpus_file = home + "/data/datasets/ptb/ptb.hdf5"

batch_size = 100
num_workers = 8

corpus_hdf5 = h5py.File(corpus_file, mode='r')
data = corpus_hdf5["validation"]

print("reading from hdf5 dataset")


# ngram generator
def get_ngrams():
    for n_gram in chunk_it(data, chunk_size=batch_size * 200):
        yield n_gram


print("Reading Line by line")
start = time.time()
for i in tqdm(range(len(data))):
    sample = data[i]
duration = time.time() - start
print("took {s} seconds to read line by line".format(s=duration))
time.sleep(0.01)

print("Reading with chunk it")
start = time.time()
for line in tqdm(get_ngrams(), total=len(data)):
    sample = line
duration = time.time() - start
print("took {s} seconds to read with chunk it".format(s=duration))
time.sleep(0.02)

print("Reading with Tensorflow dataset API")
ds = tf.data.Dataset.from_generator(get_ngrams, tf.int64)
ds = ds.prefetch(batch_size * 50)
# we call get next until we nrp out of next
ngram = ds.make_one_shot_iterator().get_next()
progress = tqdm(total=len(data))
start = time.time()
with tf.Session() as sess:
    while True:
        try:
            sample = sess.run(ngram)
            progress.update()
        except tf.errors.OutOfRangeError:
            break
progress.close()
duration = time.time() - start
print("took {s} seconds to read with tensorflow dataset API".format(s=duration))
time.sleep(0.02)

ngrams = chunk_it(data, chunk_size=batch_size * 100)
inputs = tf.placeholder(dtype=tf.int64, shape=[None, 4])
start = time.time()
progress = tqdm(total=len(data))
with tf.Session() as sess:
    for i in range(len(data)):
        sample = sess.run(inputs, feed_dict={inputs: [next(ngrams)]})
        progress.update()
progress.close()
duration = time.time() - start
print("took {s} seconds to read with tensorflow feed_dict and chunk it".format(s=duration))
time.sleep(0.02)
