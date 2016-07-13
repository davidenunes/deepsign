#!/usr/bin/env python

import time
import h5py
import os.path
from spacy.en import English

from deepsign.utils.views import sliding_windows as sliding
from deepsign.rp.index import SignIndex
from deepsign.rp.ri import RandomIndexGenerator
from deepsign.rp.encode import to_bow

import numpy as np
from numpy.linalg import svd

# *************************
# model parameters
# *************************
window_size = 4
ri_dim = 500
ri_active = 5

num_sentences = 4



data_path = "/data/datasets/wacky.hdf5"
home = os.getenv("HOME")
dataset_path = home+data_path

print(os.path.isfile(dataset_path))
print("reading hdf5 file: ", dataset_path)
dataset_name = "ukwac_sentences"

# open hdf5 file and get the dataset
f = h5py.File(dataset_path,'r')
dataset = f[dataset_name]
# do something with the dataset


print("Loading Spacy English Model")
t0 = time.time()
# load tokenizer only
nlp =  English(entity=False,load_vectors=False,parser=False,tagger=False)
t1 = time.time()
print("Done: {0:.2f} secs ".format(t1-t0))


# Create RI Index
ri_gen = RandomIndexGenerator(dim=ri_dim,active=ri_active)
sign_index = SignIndex(ri_gen)

# Co-occurrences
num_sentences = min(num_sentences,len(dataset))
occurrences = dict()
for i in range(num_sentences):
    sentence = dataset[i][0]

    t0 = time.time()
    p_sentence = nlp(sentence)
    tokens = [t.orth_ for t in p_sentence if not t.is_punct]

    # Add Tokens to ri Index
    for token in tokens:
        sign_index.add(token)

    # Compute Sliding Windows
    s_windows = sliding(tokens,window_size=window_size)

    # Encode each window as a bag-of-words and add to occurrences
    for window in s_windows:
        bow_vector = to_bow(window, sign_index)
        sign_id = sign_index.get_id(window.target)

        if sign_id not in occurrences:
            occurrences[sign_id] = bow_vector
        else:
            current_vector = occurrences[sign_id]
            occurrences[sign_id] = bow_vector + current_vector

    t1 = time.time()
    print("Sentence %d processed in: {0:.10f} secs ".format(i+1,t1 - t0))


# Create Random Indexing Occurrence Matrix
num_rows = len(occurrences)

keys = occurrences.keys()
occurr_matrix = np.matrix([occurrences[k] for k in keys])





# Perform Singular Value Decomposition
U, S, V = np.linalg.svd(occurr_matrix, full_matrices=True)
print((U.shape, S.shape, V.shape))

n_components = 100
rU = U[:, :n_components]
rS = S[:n_components]
rV = V[:n_components, :]

print((rU.shape, rS.shape, rV.shape))
# reconstruct matrix from the first rn components
r_occurr_matrix  = np.dot(rU, np.dot(rS, rV.T))
print(r_occurr_matrix.shape)



f.close()
