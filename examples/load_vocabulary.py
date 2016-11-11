import os
import h5py
from deepsign.utils.views import chunk_it
from deepsign.rp.index import SignIndex
import marisa_trie
from deepsign.rp.ri import RandomIndexGenerator
from tqdm import tqdm

home = os.getenv("HOME")
result_path = home+"/data/results/"
vocab_fname = "wacky_vocabulary.hdf5"
vocab_file = result_path + vocab_fname

h5v = h5py.File(vocab_file, 'r')


vocabulary = h5v["vocabulary"]
frequencies = h5v["frequencies"]

print(vocabulary[1:10])
print(frequencies[1:10])
print("unique words: ", len(vocabulary))

words = chunk_it(vocabulary,len(vocabulary),chunk_size=1000)
ri_gen = RandomIndexGenerator(dim=1000, active=10)
sign_index = SignIndex(ri_gen)

trie = marisa_trie.Trie(words)
print("banana" in trie)
print(trie.key_id("banana"))
print(trie.restore_key(148960))


for w in trie.keys(u'bana'):
    print(w)

print("pimba" in trie)



#index = SignIndex
#for w in tqdm(words, total=len(vocabulary)):
#    pass
    #sign_index.add(w)

#print("signs: ", len(sign_index))
#print("contains explained? ", sign_index.contains("explained"))

h5v.close()
