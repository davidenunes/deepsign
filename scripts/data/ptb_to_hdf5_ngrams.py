import os
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter

from deepsign.data.corpora.ptb import PTBReader
from deepsign.data.views import window_it, flatten_it

parser = argparse.ArgumentParser(description="PTB n-grams to hdf5")
parser.add_argument('-n', dest="n", type=int, default=4)
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME") + "/data/datasets/ptb")
parser.add_argument('-out_dir', dest="out_path", type=str, default=os.getenv("HOME") + "/data/results/")
parser.add_argument('-flatten', dest="flatten", type=bool, default=False)
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

hdf5_path = os.path.join(args.data_dir, "ptb_{}.hdf5".format(args.n))
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
    # if flatten, ngrams go across sentences
    if args.flatten:
        word_it = flatten_it(corpus_stream)
        ngrams = window_it(word_it, args.n)
    else:
        sentence_ngrams = (window_it(sentence, args.n) for sentence in corpus_stream)
        ngrams = flatten_it(sentence_ngrams)

    ngram_ids = [list(map(lambda w: vocab[w], ngram)) for ngram in ngrams]
    ngrams = np.array(ngram_ids)
    dataset = hdf5_file.create_dataset(name, data=ngrams, compression="gzip")
    dataset.attrs['n'] = args.n


try:
    print("storing training set")
    training_dataset = ptb_reader.training_set()
    store_ngrams(training_dataset, "training")

    print("storing test set")
    test_dataset = ptb_reader.test_set()
    store_ngrams(test_dataset, "test")

    print("storing validation set")
    validation_dataset = ptb_reader.validation_set()
    store_ngrams(validation_dataset, "validation")

    hdf5_file.close()
    print("done")
except (KeyboardInterrupt, Exception) as e:
    os.remove(hdf5_path)
    raise e
