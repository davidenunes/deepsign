#!/usr/bin/env python
import h5py
import os.path
import numpy as np

home = os.getenv("HOME")
result_path = home+"/data/datasets/"
vocab_h5 = result_path + "wacky_vocab_10M.hdf5"
print("Reading from ", vocab_h5)
h5f = h5py.File(vocab_h5, 'r')

vocab = h5f["vocabulary"]
freq = h5f["frequencies"]

total_freq = np.sum(freq)
print("Total Words: %s" % format(total_freq, ",d"))
print("Unique Words: %s" % format(len(vocab), ",d"))
print("Top 10: %s" % str(vocab[0:10]))

freq_threshold = 1000

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
