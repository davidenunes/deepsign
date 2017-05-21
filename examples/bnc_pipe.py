import os

import h5py
from tqdm import tqdm

from deepsign.io.corpora.pipe import BNCPipe
from deepsign.rp.encode import to_bow
from deepsign.rp.index import SignIndex, Generator
from deepsign.utils.views import chunk_it, sliding_windows

home = os.getenv("HOME")

data_dir = home + "/data/datasets/"
corpus_file = data_dir + "bnc.hdf5"

corpus_hdf5 = h5py.File(corpus_file, 'r')
corpus_dataset = corpus_hdf5["sentences"]



n_rows = 1000
sentences = chunk_it(corpus_dataset, n_rows=n_rows, chunk_size=100000)
pipeline = BNCPipe(datagen=sentences,lemmas=True)

ri_gen = Generator(1000,10)
index = SignIndex(ri_gen)


for s in tqdm(pipeline,total=n_rows):
    index.add_all(s)


    windows = sliding_windows(s,window_size=2)

    for window in windows:
        pass
        #words = window.left + window.right
        #ris = [index.get_ri(word).to_vector() for word in words]
        bow = to_bow(window,index,include_target=False,normalise=True)