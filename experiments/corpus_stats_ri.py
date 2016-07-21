#!/usr/bin/env python

import time
import h5py
import os.path
from spacy.en import English

from deepsign.utils.views import sliding_windows as sliding
from deepsign.rp.index import SignIndex
from deepsign.rp.ri import RandomIndexGenerator
from deepsign.rp.encode import to_bow

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np

home = os.getenv("HOME")
result_path = home+"/data/results/"
filename = "random_indexing.hdf5"
corpus_file = result_path + filename
print("Reading from ", corpus_file)

ri_dim = 600
ri_active = 4


vocabulary_dset = "vocabulary"
frequencies_dset = "frequencies"
ri_dset = "ri_d{0}_a{1}".format(ri_dim,ri_active)
ri_sum_dset = ri_dset+"_sum"

h5f = h5py.File(corpus_file, 'r')

vocabulary = h5f[vocabulary_dset]
frequencies = h5f[frequencies_dset]
ri = h5f[ri_dset]
vectors = h5f[ri_sum_dset]

print("Number of unique words: ",len(vocabulary))

# sort word frequencies


# sort frequencies from high to low
sorted_idx = np.argsort(frequencies)[::-1]
sorted_freq = frequencies[:][sorted_idx]
sorted_vocab = vocabulary[:][sorted_idx]

freq_cut = 40

print(sorted_freq[range(freq_cut)])
print(sorted_vocab[range(freq_cut)])


words = sorted_vocab[1:freq_cut]
freq = sorted_freq[1:freq_cut]
x_values = np.arange(1,len(words)+1,1)


plt.bar(x_values, freq, align='center',alpha=0.8,color="#2aa198", edgecolor="#2aa198")

plt.xticks(x_values, words,rotation='vertical')
plt.ylabel("Word Frequency")

# Remove the plot frame lines. They are unnecessary chartjunk.
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)


pp = PdfPages('frequencies.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.clf()

#sorted_freq = frequencies[sorted_idx]
#sorted_vocab = vocabulary[sorted_idx]



#for i in range(10):
#    print("{0} : {1}".format(sorted_vocab[i],sorted_freq[i]))



h5f.close()