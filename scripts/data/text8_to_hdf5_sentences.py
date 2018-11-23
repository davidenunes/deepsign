import os
import numpy as np
import h5py
import argparse
import marisa_trie
from collections import Counter
from tqdm import tqdm
from deepsign.nlp import is_token

from deepsign.data.corpora.text8 import Text8Corpus
from deepsign.data import iterators

parser = argparse.ArgumentParser(description="text8 n-grams to hdf5")
parser.add_argument('-n', dest="n", type=int, default=4)
parser.add_argument('-data_dir', dest="data_dir", type=str, default=os.getenv("HOME") + "/data/datasets/text8")
parser.add_argument('-out_filename', dest="out_filename", type=str, default="text8")
parser.add_argument('-vocab_limit', dest="vocab_limit", type=int, default=70000)
parser.add_argument('-unk_token', dest="unk_token", type=str, default="<UNK>")

args = parser.parse_args()

dataset = os.path.join(args.data_dir, "text8.txt")

corpus = Text8Corpus(dataset, sentence_length=1000)

word_counter = Counter()
for word in tqdm(iterators.flatten_it(corpus)):
    word_counter[word] += 1

print("total words ", sum(word_counter.values()))
sorted_counts = word_counter.most_common(args.vocab_limit)
word_list, _ = zip(*sorted_counts)

vocab = marisa_trie.Trie(word_list)
print("vocab size: ", len(vocab))

new_counter = Counter()
for word in word_counter.keys():
    if word in vocab:
        new_counter[word] += word_counter[word]
    if word not in vocab:
        new_counter[args.unk_token] += word_counter[word]

word_counter = new_counter
sorted_counts = word_counter.most_common(args.vocab_limit)
word_list, word_freq = zip(*sorted_counts)
vocab = marisa_trie.Trie(word_list)

print("vocab size: ", len(vocab))

vocabulary = np.array(word_list, dtype="S")
ids = np.array([vocab[word] for word in word_list])
frequencies = np.array(word_freq)

hdf5_path = os.path.join(args.data_dir, args.out_filename + "_{}.hdf5".format(args.n))
hdf5_file = h5py.File(hdf5_path, "a")

# write words (needs str type)
h5str_dtype = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset("vocabulary", data=vocabulary, dtype=h5str_dtype, compression="gzip")
hdf5_file.create_dataset("frequencies", data=frequencies, compression="gzip")
hdf5_file.create_dataset("ids", data=ids, compression="gzip")
print("vocabulary and frequencies written")
print("processing n-grams...")

filtered_corpus = map(lambda w: w if w in vocab else args.unk_token, iterators.flatten_it(corpus))

total_words = sum(word_freq)
n_training = int(total_words * 0.8)

# leave out
n_leave = total_words - n_training

n_eval = n_leave // 2
n_test = n_leave - n_eval

# 80 / 10 / 10 split
ngrams = iterators.window_it(filtered_corpus, args.n)

# TODO iterative: 1 consume n n grams 2 extend hdf5 dataset 3 write to dataset
# https://stackoverflow.com/questions/34531479/writing-a-large-hdf5-dataset-using-h5py
# I also have some old examples with wacky corpus

#ngram_ids = [list(map(lambda w: vocab[w], ngram)) for ngram in ngrams]
#ngrams = np.array(ngram_ids)
#dataset = hdf5_file.create_dataset("full", data=ngrams, compression="gzip")
#dataset.attrs['n'] = args.n

# for ngram in views.consume_it(ngrams):
#    print(ngram)


hdf5_file.close()
