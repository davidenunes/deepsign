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
ri_dim = 400
ri_active = 4




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


num_sentences = 100000 #len(dataset)
# Co-occurrences
num_sentences = min(num_sentences,len(dataset))
occurrences = dict()
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
        else:
            current_vector = occurrences[sign_id]
            occurrences[sign_id] = bow_vector + current_vector

    t1 = time.time()
    print("Sentence {0} processed in: {1:.10f} secs ".format(i+1,t1 - t0))



# get rid of spacy
nlp = None
gc.collect()

# Create corpus matrix based on ri
num_rows = len(occurrences)

keys = occurrences.keys()
c_matrix = [occurrences[key] for key in keys]

# squash each ri vector to [-1,1]
c_matrix = [v/np.max(v,axis=0) for v in c_matrix]

# apply tanh transformation to smooth out the different patterns
c_matrix = np.matrix([np.tanh(v) for v in c_matrix])

c_matrix = np.matrix(c_matrix)

print(c_matrix.shape)

# Perform Singular Value Decomposition
U, s, V_t = np.linalg.svd(c_matrix, full_matrices=False)
print((U.shape, s.shape, V_t.shape))


print("Printing a 2D space with SVD-2 vectors")
# number of components to consider
k = 2
rU = U[:, :k]
rs = s[:k]


print((rU.shape, rs.shape))
ld_matrix = np.dot(rU, np.diag(rs))
print(ld_matrix.shape)

# get sample only
sample_indices = np.random.choice(ld_matrix.shape[0], 1000)
samples = ld_matrix[sample_indices,:]
print(samples.shape)


# word labels
words = np.array([sign_index.get_sign(key) for key in keys])
words = words[sample_indices]
print("Words : ",len(words))
print(words[range(5)])

plt.scatter(samples[:, 0],samples[:, 1])

for i,word in enumerate(words):
    plt.annotate(word,(samples[i,0],samples[i,1]))


pp = PdfPages('svd.pdf')
plt.savefig(pp, format='pdf')
#plt.savefig("svd.png")
pp.close()
plt.clf()


print("Printing a t-SNE space with SVD-10 vectors")
# number of components to consider
k = 10
rU = U[:, :k]
rs = s[:k]


print((rU.shape, rs.shape))
ld_matrix = np.dot(rU, np.diag(rs))
print(ld_matrix.shape)


# get sample only
samples = ld_matrix[sample_indices,:]
print(samples.shape)

model = TSNE(n_components=2, random_state=0,method='barnes_hut',n_iter=500,learning_rate=800,angle=0.6)
np.set_printoptions(suppress=True)
embedded_space = model.fit_transform(samples)

plt.scatter(embedded_space[:, 0],embedded_space[:, 1])

for i,word in enumerate(words):
    plt.annotate(word,(embedded_space[i,0],embedded_space[i,1]))


#plt.savefig("t-sne_svd10.png")

pp = PdfPages('t-sne_svd10.pdf')
plt.savefig(pp, format='pdf')
pp.close()



f.close()
