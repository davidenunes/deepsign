import os
import h5py
#from deepsign.utils.views import chunk_it
from deepsign.rp.index import TrieSignIndex
import marisa_trie
from deepsign.rp.ri import Generator
from tqdm import tqdm
#import numpy as np
import time

home = os.getenv("HOME")
result_path = home+"/data/gold_standards/"
vocab_fname = "wacky_vocab_1M.hdf5"
vocab_file = result_path + vocab_fname

h5v = h5py.File(vocab_file, 'r')


vocabulary = h5v["vocabulary"]
frequencies = h5v["frequencies"]

print("unique words: ", len(vocabulary))

#words = chunk_it(vocabulary, len(vocabulary), chunk_size=1000)

print("loading vocab")
t0 = time.time()
trie = marisa_trie.Trie(list(vocabulary))
t1 = time.time()
print("vocab loaded")
print(t1-t0)

top10w = list(vocabulary[0:10])
top10f = list(frequencies[0:10])
top10ids = [trie.get(top10w[i]) for i in range(10)]
top10w_trie = [trie.restore_key(i) for i in top10ids]

print(top10w)
print(top10f)
print(top10w_trie)

ri_gen = Generator(dim=1000, num_active=10)

t0 = time.time()
sign_index = TrieSignIndex(ri_gen, list(vocabulary[:]))
t1 = time.time()
print(t1-t0)

print(top10ids)
top10w_index = [sign_index.get_sign(i) for i in top10ids]
print(top10w_index)


#test load top ten
print("=============================================")
index = TrieSignIndex(generator=ri_gen, vocabulary=top10w)
print(top10w)
top10ids = [index.get_id(w) for w in top10w]
print(top10ids)
freq = TrieSignIndex.map_frequencies(top10w,top10f,index)
top10freq = [freq[i] for i in top10ids]
print(top10freq)


h5v.close()
