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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from sklearn.manifold import TSNE

import gc

# *************************
# model parameters
# *************************
window_size = 4
ri_dim = 1000
ri_active = 10


home = os.getenv("HOME")
corpus_file = "/data/datasets/wacky.hdf5"
result_path = home+"/data/results"
corpus_file = home + corpus_file
print(os.path.isfile(corpus_file))

print("reading hdf5 file: ", corpus_file)
dataset_name = "ukwac_sentences"

# open hdf5 file and get the dataset
h5f = h5py.File(corpus_file, 'r')
dataset = h5f[dataset_name]
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


num_sentences = 100 #len(dataset)
# Co-occurrences
num_sentences = min(num_sentences,len(dataset))
occurrences = dict()
frequencies = dict()

for i in range(num_sentences):
    sentence = dataset[i][0]

    t0 = time.time()
    p_sentence = nlp(sentence)

    # remove punctuation and stop words
    tokens = [t.orth_ for t in p_sentence if not t.is_punct and not t.is_stop]
    #print(p_sentence)
    #print(tokens)

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
            frequencies[sign_id] = 1
        else:
            current_vector = occurrences[sign_id]
            occurrences[sign_id] = bow_vector + current_vector
            frequencies[sign_id] += 1

    t1 = time.time()
    print("Sentence {0} processed in: {1:.10f} secs ".format(i+1,t1 - t0))

h5f.close()

# open hdf5 file to write

# prepare data
word_ids = occurrences.keys()
vocabulary = np.array([sign_index.get_sign(w_id).encode("utf8") for w_id in word_ids])
frequencies = np.array([frequencies[w_id]] for w_id in word_ids)
ri_vectors = np.array([sign_index.get_ri(w).to_vector() for w in vocabulary])
ri_avg_vectors = [occurrences[w_id] for w_id in word_ids]

filename = "random_indexes.hdf5"
corpus_file = result_path + filename
print("writing to ", corpus_file)

dataset_name = "ri_d{0}_a{1}".format(ri_dim,ri_active)
print("dataset: "+dataset_name)

h5f = h5py.File(corpus_file, 'w')
dt = h5py.special_dtype(vlen=str)

vocabulary_data = h5f.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
print("vocabulary data written")

count_data = h5f.create_dataset("frequencies", data=frequencies, compression="gzip")
print("count data written")

ri_data = h5f.create_dataset(dataset_name, data=ri_vectors, compression="gzip")
print("random index vectors written")

sum_vectors = h5f.create_dataset(dataset_name+"_sum", data=ri_avg_vectors, compression="gzip")
print("random index sum vectors written")

h5f.close()






