#!/usr/bin/env python
import h5py
import os.path
import numpy as np

home = os.getenv("HOME")
result_path = home+"/data/gold_standards/"
corpus_h5 = result_path + "bnc.hdf5"
vocab_h5 = result_path + "bnc_vocab_lemma.hdf5"

#corpus_h5 = result_path + "bnc_full.hdf5"
#vocab_h5 = result_path + "bnc_vocab.hdf5"

print("Reading from ", vocab_h5)
h5f = h5py.File(vocab_h5, 'r')

h5corpus = h5py.File(corpus_h5,'r')

vocab = h5f["vocabulary"]
freq = h5f["frequencies"]

total_freq = np.sum(freq)
print("Total Sentences: %s" % format(len(h5corpus["sentences"]), ",d"))
print("Total Words: %s" % format(total_freq, ",d"))
print("Unique Words: %s" % format(len(vocab), ",d"))
print("Top 10: %s" % str(vocab[0:10]))

print("Last 10: %s" % str(vocab[-11:-1]))

# frequency thresholding could be useful
freq_threshold = 447
if freq_threshold > 0:
    indexes = np.where(freq[:] >= freq_threshold)[0]
    indexes = slice(0,len(indexes),1)
    print(indexes)
    freq_cut = freq[indexes]
    print("Last: ",(vocab[indexes.stop-1],freq[indexes.stop-1]))
    total_freq = np.sum(freq_cut)
    print("Frequency cut threshold set to %d" % freq_threshold)
    print("Total Words: %s" % format(total_freq, ",d"))
    print("Unique Words: %s" % format(len(freq_cut), ",d"))



h5f.close()
h5corpus.close()
