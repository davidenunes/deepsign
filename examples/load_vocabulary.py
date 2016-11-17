import os
import h5py
from deepsign.utils.views import chunk_it
from deepsign.rp.index import TrieSignIndex
import marisa_trie
from deepsign.rp.ri import RandomIndexGenerator
from tqdm import tqdm
import numpy as np
import time

home = os.getenv("HOME")
result_path = home+"/data/results/"
vocab_fname = "wacky_vocab_1M.hdf5"
vocab_file = result_path + vocab_fname

h5v = h5py.File(vocab_file, 'r')


vocabulary = h5v["vocabulary"]
frequencies = h5v["frequencies"]

print(vocabulary[1:10])
print(frequencies[1:10])
print("unique words: ", len(vocabulary))

words = chunk_it(vocabulary, len(vocabulary), chunk_size=1000)


print(np.array(vocabulary[1:2]))
print()

t0 = time.time()
trie = marisa_trie.Trie(list(vocabulary[()]), weights=list(frequencies[()]))
t1 = time.time()
print(t1-t0)


ri_gen = RandomIndexGenerator(dim=1000, active=10)

t0 = time.time()
sign_index = TrieSignIndex(ri_gen, list(vocabulary[()]))
t1 = time.time()
print(t1-t0)



#index = SignIndex
#for w in tqdm(words, total=len(vocabulary)):
#    pass
    #sign_index.add(w)

#print("signs: ", len(sign_index))
#print("contains explained? ", sign_index.contains("explained"))

h5v.close()
