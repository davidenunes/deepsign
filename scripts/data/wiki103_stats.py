import os
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter
from tqdm import tqdm

from deepsign.data.corpora.wiki103 import WikiText103
from deepsign.data.iterators import window_it, flatten_it

parser = argparse.ArgumentParser(description="wikitext 103 n-grams to hdf5")
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME") + "/data/datasets/wikitext-103")
parser.add_argument('-mark_eos', dest="mark_eos", type=bool, default=True)

args = parser.parse_args()

# ======================================================================================
# Build Vocabulary
# the dataset is quite small so we read everything first to build the vocab
# we then use that vocabulary to encode each n-gram
# ======================================================================================
corpus_reader = WikiText103(args.data_dir, args.mark_eos)

# load vocabulary
word_counter = Counter()
for words in tqdm(corpus_reader.full()):
    word_counter.update(words)

vocab = marisa_trie.Trie(word_counter.keys())
sorted_counts = word_counter.most_common()
word_list, word_freq = zip(*sorted_counts)

u_word_list = [word.encode("utf-8") for word in word_list]

# vocabulary = np.array([freq[i][0].encode("utf8") for word in word_list])
# encode strings in array 0 terminated bytes
vocabulary = np.array(u_word_list, dtype="S")
ids = np.array([vocab[word] for word in word_list])
frequencies = np.array(word_freq)

hdf5_path = os.path.join(args.data_dir, "wiki103_stats.hdf5")
try:
    os.remove(hdf5_path)
except OSError:
    pass
hdf5_file = h5py.File(hdf5_path, "a")

# write words (needs str type)
h5str_dtype = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset("vocabulary", data=vocabulary, dtype=h5str_dtype)
hdf5_file.create_dataset("frequencies", data=frequencies)
hdf5_file.create_dataset("ids", data=ids)
print("vocabulary and frequencies written")
print("done")
