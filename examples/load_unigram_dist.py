import marisa_trie
import h5py
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import fixed_unigram_candidate_sampler as unigram_sampler

ngram_size = 4
corpus_path = os.path.join(os.getenv("HOME"), "data/datasets/ptb/")

corpus = h5py.File(os.path.join(corpus_path, "ptb_{}.hdf5".format(ngram_size)), mode='r')
vocab = marisa_trie.Trie(corpus["vocabulary"])

# since the vocab doesn't preserve the id order
ids = corpus["ids"]
freq = corpus["frequencies"]
unigram = freq / np.sum(freq)

# order by ids if the model uses the vocab ids
ids, prob = zip(*sorted(zip(ids, unigram)))

idmap = {ids[i]: i for i in range(len(ids))}


with tf.Session() as ss:
    sample = ss.run(unigram_sampler([[26]], 1, 5, True, len(vocab), unigrams=prob))
    print([vocab.restore_key(i) for i in sample.sampled_candidates])
