#!/usr/bin/env python

import time
import h5py
import os.path
from spacy.en import English
from deepsign.utils.views import sliding_windows
from deepsign.rp import encode as enc


data_path = "/data/datasets/wacky.hdf5"
home = os.getenv("HOME")
dataset_path = home+data_path

print(os.path.isfile(dataset_path))
print("reading file: ", dataset_path)

dataset_name = "ukwac_sentences"

# open hdf5 file and get the dataset
f = h5py.File(dataset_path,'r')
dataset = f[dataset_name]
# do something with the dataset


print("Loading Spacy English Model")
t0 = time.time()
nlp =  English()
t1 = time.time()
print("Done: %d secs ",t1-t0 / 1000)


"""
num_sentences = 10
for i in range(num_sentences):
    sentence = dataset[i][0]

    tokens1 = tk.tokenize(sentence)
    print(tokens1)


    # spacy process
    doc = nlp(sentence)
    tokens2 = [w.orth_ for w in doc]
    print(tokens2)

    tokens = [t.orth_ for t in doc if not t.is_punct]


    #windows = sliding_windows(tokens,window_size=2)
    #for w in windows:
    #    print(w)

    #print dep tree
    for token in doc:
        # print head
        print(token.dep_ + "(" + token.head.orth_ + "("+token.head.pos_+")" + "," + token.orth_ + "("+token.pos_+")"+ ")")

"""

f.close()
