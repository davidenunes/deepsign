#!/usr/bin/env python

import h5py
import os.path
from deepsign.utils.views import sliding_windows
from deepsign.rp import encode as enc
import deepsign.text.tokenizer as tk
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


#sentence = dataset[6][0]
sentence = ":) Granada don't, the 'Alhambra' and the city 's plain , the Alhama mountain , the picturesque Axarquía region , Antequera and its stunning torcal mountain , the El Chorro gorge , Ronda mountain , Cortes Nature Reserve , Alcornocales mountain , magnificent view out to Africa from Tarifa , these are just some of the place , people and pleasure that you will discover as you ride the ancient trail of Andalusia ."

print(sentence)

tokens1 = tk.tokenize(sentence)
print(tokens1)

doc = nlp(sentence)
tokens2 = [w.orth_ for w in doc]
print(tokens2)

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
