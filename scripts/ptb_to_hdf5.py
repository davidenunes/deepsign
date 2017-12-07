import sys
import os
from tqdm import tqdm
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter

from deepsign.io.corpora.ptb import PTBReader
from deepsign.utils.views import ngram_windows
from deepsign.utils import h5utils

parser = argparse.ArgumentParser(description="PTB n-grams to hdf5")
parser.add_argument('-n', dest="n", type=int, default=4)
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME") + "/data/gold_standards/ptb")
parser.add_argument('-out_dir', dest="out_path", type=str, default=os.getenv("HOME") + "/data/gold_standards")
parser.add_argument('-out_filename', dest="out_filename", type=str, default="ptb.hdf5")
args = parser.parse_args()

# ======================================================================================
# Build Vocabulary
# the dataset is quite small so we read everything first to build the vocab
# we then use that vocabulary to encode each n-gram
# ======================================================================================
ptb_reader = PTBReader(args.data_dir)

# load vocabulary
word_counter = Counter()
for words in ptb_reader.full():
    word_counter.update(words)

vocab = marisa_trie.Trie(word_counter.keys())
sorted_counts = word_counter.most_common()
word_list, word_freq = zip(*sorted_counts)

# vocabulary = np.array([freq[i][0].encode("utf8") for word in word_list])
# encode strings in array 0 terminated bytes
vocabulary = np.array(word_list, dtype="S")
ids = np.array([vocab[word] for word in word_list])
frequencies = np.array(word_freq)

hdf5_path = os.path.join(args.data_dir, args.out_filename)
hdf5_file = h5py.File(hdf5_path, "a")

# write words (needs str type)
h5str_dtype = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset("vocabulary", data=vocabulary, dtype=h5str_dtype, compression="gzip")
hdf5_file.create_dataset("frequencies", data=frequencies, compression="gzip")
hdf5_file.create_dataset("ids", data=ids, compression="gzip")
print("vocabulary and frequencies written")
print("processing n-grams...")

# ======================================================================================
#   Write n-grams
# ======================================================================================


def store_ngrams(corpus_stream, name):
    sentence_ngrams = (ngram_windows(sentence, args.n) for sentence in corpus_stream)
    ngrams = (ngram for ngrams in sentence_ngrams for ngram in ngrams)
    n_gram_ids = [list(map(lambda w: vocab[w], ngram)) for ngram in ngrams]
    ngrams = np.array(n_gram_ids)
    # sample = hdf5_file["ngrams/training"][100:110]
    # sample = [list(map(lambda id: vocab.restore_key(id), ngram)) for ngram in sample]
    dataset = hdf5_file.create_dataset(name, data=ngrams, compression="gzip")
    dataset.attrs['n'] = args.n


training_dataset = ptb_reader.training_set()
store_ngrams(training_dataset, "training")

test_dataset = ptb_reader.test_set()
store_ngrams(test_dataset, "test")

validation_dataset = ptb_reader.validation_set()
store_ngrams(validation_dataset, "validation")


hdf5_file.close()
print("done")
