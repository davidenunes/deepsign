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
from tqdm import tqdm


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE

import gc

# *************************
# model parameters
# *************************
window_size = 3
ri_dim = 600
ri_active = 4


home = os.getenv("HOME")
corpus_file = "/data/datasets/wacky.hdf5"
result_path = home+"/data/results/"
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
nlp =  English(entity=False,load_vectors=False,parser=True,tagger=True)
t1 = time.time()
print("Done: {0:.2f} secs ".format(t1-t0))


# Create RI Index
ri_gen = RandomIndexGenerator(dim=ri_dim,active=ri_active)
sign_index = SignIndex(ri_gen)

# process 1 million sentences
num_sentences = 500000#len(dataset)
# Co-occurrences
num_sentences = min(num_sentences,len(dataset))
occurrences = dict()
frequencies = dict()

for i in tqdm(range(num_sentences)):
    sentence = dataset[i][0]

    t0 = time.time()
    p_sentence = nlp(sentence)

    # remove punctuation and stop words
    # TODO additional pre-processing substitute time and URLs by T_TIME, and T_URL, etc?
    # TODO all caps to lower except entities
    # TODO remove useless tokens from the previously process @card@, @ord@, (check what tokens are considered in wacky)
    # TODO
    # TODO substitute numbers for T_NUMBER ?



    def get_string(token):
        w = token.orth_

        # some words are tokenised with 's and n't, apply this before filtering stop words
        if w == "'s":
            w = "is"
        elif w == "n't":
            w = "not"
        return w



    tokens = [t.orth_ for t in p_sentence if not t.is_punct]


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

h5f.close()

# open hdf5 file to write

# TODO do this online it is too memory intensive

# prepare data
word_ids = occurrences.keys()
print("processing {0} word vectors".format(len(word_ids)))

vocabulary = np.array([sign_index.get_sign(w_id).encode("UTF-8") for w_id in word_ids])
frequencies = np.array([frequencies[w_id] for w_id in word_ids])
ri_vectors = np.array([sign_index.get_ri(w.decode("UTF-8")).to_sparse() for w in vocabulary])
ri_avg_vectors = [occurrences[w_id] for w_id in word_ids]

filename = "random_indexing.hdf5"
corpus_file = result_path + filename
print("writing to ", corpus_file)

dataset_name = "ri_d{0}_a{1}".format(ri_dim,ri_active)
print("dataset: "+dataset_name)

h5f = h5py.File(corpus_file, 'w')
dt = h5py.special_dtype(vlen=str)

# the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
vocabulary_data = h5f.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
print("vocabulary data written")

count_data = h5f.create_dataset("frequencies", data=frequencies, compression="gzip")
print("count data written")

# random indexing vectors are stored in sparse mode (active indexes only),
# reconstruct using ri.from_sparse(dim,active,active_list) to get a RandomIndex object
ri_data = h5f.create_dataset(dataset_name, data=ri_vectors, compression="gzip")
print("random index vectors written")
ri_data.attrs["dimension"] = ri_dim
ri_data.attrs["active"] = ri_active

# avg are stored as is since there is no guarantee that these will be sparse (depends on the params)
sum_vectors = h5f.create_dataset(dataset_name+"_sum", data=ri_avg_vectors, compression="gzip")
print("random index sum vectors written")


h5f.close()






