#!/usr/bin/env python

import h5py
import os.path
from deepsign.utils.views import sliding_windows
from deepsign.rp import encode as enc
from spacy.en import English

data_path = "/Dropbox/research/Data/WaCKy/wacky.hdf5"
home = os.getenv("HOME")
dataset_path = home+data_path

print(os.path.isfile(dataset_path))
print("reading file: ", dataset_path)

dataset_name = "ukwac_sentences"

# open hdf5 file and get the dataset
f = h5py.File(dataset_path,'r')
dataset = f[dataset_name]
# do something with the dataset


print("Loading English Model...")
nlp =  English()
print("Done!")

num_sentences = 100
for i in range(num_sentences):
    sentence = dataset[i][0]
    doc = nlp(sentence)
    print(doc)

    tokens = [t.orth_ for t in doc if not t.is_punct]
    tokens_ents = [t.orth_ for t in doc if t.ent_type != 0]
    print("entities:")
    print(tokens_ents)

    #windows = sliding_windows(tokens,window_size=2)
    #for w in windows:
    #    print(w)

    #print dep tree
    for token in doc:
        # print head
        print(token.dep_ + "(" + token.head.orth_ + "("+token.head.pos_+")" + "," + token.orth_ + "("+token.pos_+")"+ ")")
        for child in token.children:
            print(child.dep_ + "(" + token.orth_ + "("+token.pos_+")" + "," + child.orth_ + "("+child.pos_+")" +  ")")

f.close()
