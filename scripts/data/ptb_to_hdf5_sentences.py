import os
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter

from deepsign.data.corpora.ptb import PTBReader
from deepsign.data import views

parser = argparse.ArgumentParser(description="PTB n-grams to hdf5")
parser.add_argument('-n', dest="n", type=int, default=4)
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME") + "/data/datasets/ptb")
parser.add_argument('-out_dir', dest="out_path", type=str, default=os.getenv("HOME") + "/data/results/")
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


def write_vocab(vocabulary_ids, vocabulary_data, frequency_data):
    # write words (needs str type)
    h5str_dtype = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset("vocabulary", data=vocabulary_data, dtype=h5str_dtype, compression="gzip")
    hdf5_file.create_dataset("frequencies", data=frequency_data, compression="gzip")
    hdf5_file.create_dataset("ids", data=vocabulary_ids, compression="gzip")
    print("vocabulary and frequencies written")


# ======================================================================================
#   Write sentences
# ======================================================================================


def store_sentences(corpus_stream, name):
    sentences = (" ".join(sentence) for sentence in corpus_stream)
    sentences = np.array(list(sentences), dtype="S")

    h5str_dtype = h5py.special_dtype(vlen=str)
    hdf5_file.create_dataset(name, data=sentences, dtype=h5str_dtype, compression="gzip")
    # dataset.attrs['n'] = args.n


print("writing vocab")
write_vocab(ids, vocabulary, frequencies)

print("processing n-grams...")

training_dataset = ptb_reader.training_set()
store_sentences(training_dataset, "training")

test_dataset = ptb_reader.test_set()
store_sentences(test_dataset, "test")

validation_dataset = ptb_reader.validation_set()
store_sentences(validation_dataset, "validation")

hdf5_file.close()
print("done")
