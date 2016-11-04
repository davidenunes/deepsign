#!/usr/bin/env python

import h5py
import os.path
import numpy as np

home = os.getenv("HOME")
result_path = home+"/data/results/"
filename = "wacky_vocabulary.hdf5"

vocab_file = result_path + filename
print("Reading from ", vocab_file)

h5f = h5py.File(vocab_file, 'r')

vocabulary = h5f["vocabulary"]
frequencies = h5f["frequencies"]

print("Number of unique words: ",len(vocabulary))

print(vocabulary[1:10])
print(frequencies[1:10])



h5f.close()