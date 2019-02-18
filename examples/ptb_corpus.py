from deepsign.data.corpora.ptb import PTBReader
from tensorx.data import itertools as itx
import marisa_trie
import os
import h5py
import numpy as np

home = os.getenv("HOME")
corpus_path = os.path.join(home, 'data/datasets/ptb')

WINDOW_SIZE = 5
BATCH_SIZE = 2

ptb = PTBReader(corpus_path)

corpus_stats = h5py.File(os.path.join(corpus_path, "ptb_stats.hdf5"), mode='r')
vocab = marisa_trie.Trie(corpus_stats["vocabulary"])


def pipeline(corpus_stream,
             n_gram_size=WINDOW_SIZE,
             batch_size=BATCH_SIZE,
             shuffle=True,
             flatten=True):
    """ Corpus Pipeline.

    Args:
        n_gram_size: the size of the n-gram window
        corpus_stream: the stream of sentences of words
        batch_size: batch size for the n-gram batch
        shuffle: if true, shuffles the n-grams according to a buffer size
        flatten: if true sliding windows are applied over a stream of words rather than within each sentence
        (n-grams can cross sentence boundaries)
    """

    if flatten:
        word_it = itx.flatten_it(corpus_stream)
        n_grams = itx.window_it(word_it, n_gram_size)
    else:
        sentence_n_grams = (itx.window_it(sentence, n_gram_size) for sentence in corpus_stream)
        n_grams = itx.flatten_it(sentence_n_grams)

    # at this point this is an n_gram iterator
    n_grams = ([vocab[w] for w in ngram] for ngram in n_grams)

    if shuffle:
        n_grams = itx.shuffle_it(n_grams, BATCH_SIZE * 1000)

    n_grams = itx.batch_it(n_grams, size=batch_size, padding=False)
    return n_grams


epochs = 2
train_data = itx.repeat_apply(lambda corpus: pipeline(corpus.training_set(1)),
                              ptb,
                              n=epochs,
                              enum=True)

data_it = itx.bptt_it(itx.flatten_it(ptb.training_set()), seq_len=20,seq_prob=0.95, min_seq_len=5, batch_size=80)

for data in data_it:
    shape = np.shape(data)
    if shape[-1] <= 5:
        print(shape)
