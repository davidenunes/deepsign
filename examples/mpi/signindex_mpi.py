import os
import h5py
from deepsign.utils.views import chunk_it
from deepsign.rp.index import TrieSignIndex
import marisa_trie
from deepsign.rp.ri import Generator
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

home = os.getenv("HOME")
result_path = home+"/data/results/"
vocab_fname = "wacky_vocab_1M.hdf5"
vocab_file = result_path + vocab_fname

h5v = h5py.File(vocab_file, 'r')
vocabulary = h5v["vocabulary"]
frequencies = h5v["frequencies"]

ri_gen = Generator(dim=1000, active=10)
sign_index = TrieSignIndex(generator=ri_gen,
                            signs=list(vocabulary[()]),
                            frequencies=list(frequencies[()]))

index = mp.Value(TrieSignIndex,sign_index)


def f(index):
    print("hello" in index)

if __name__ == '__main__':



    p = mp.Process(target=f, args=index)
    p.start()
    p.join()


