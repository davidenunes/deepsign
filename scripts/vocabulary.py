#!/usr/bin/env python

import h5py
import os.path
import numpy as np
from tqdm import tqdm
from collections import Counter

from deepsign.nlp.tokenization import Tokenizer
from deepsign.utils.views import chunk_it
from experiments.pipe.wacky_pipe import WaCKyPipe


def build_vocabulary(corpus_file, output_file=None, max_sentences=0):
    input_hdf5 = h5py.File(corpus_file, 'r')
    #dataset_name = "sentences_lemmatised"
    dataset_name = "sentences"

    dataset = input_hdf5[dataset_name]

    if max_sentences > 0:
        num_sentences = min(max_sentences, len(dataset))
    else:
        num_sentences = len(dataset)

    gen = chunk_it(dataset,num_sentences,chunk_size=250)
    tokenizer = Tokenizer()
    pipe = WaCKyPipe(gen,tokenizer,filter_stop=False)
    freq = Counter()

    for tokens in tqdm(pipe, total=num_sentences):
        #tqdm.write(str(tokens))
        for token in tokens:
            normal_token = token.lower()
            freq[normal_token] += 1

    input_hdf5.close()
    tqdm.write("{0} unique words".format(len(freq)))
    # order by frequency
    freq = freq.most_common()

    for i in range(10):
        (w,f) = freq[i]
        print("{0}:{1}".format(w, f))

    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'w')
        word_ids = range(len(freq))

        # encode explicitly so that hdf5 can take an array of variable length strings and store it
        # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
        vocabulary = np.array([freq[i][0].encode("utf8") for i in range(len(freq))])

        dt = h5py.special_dtype(vlen=str)
        output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
        print("vocabulary written")

        freq = np.array([freq[i][1] for i in range(len(freq))])
        output_hdf5.create_dataset("frequencies", data=freq, compression="gzip")
        print("frequencies written")

        output_hdf5.close()
        print("done")


if __name__ == '__main__':
    # model parameters
    max_sentences = 10000

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = home + "/data/datasets/wacky_1M.hdf5"
    index_filename = home + "/data/results/wacky_vocab_1M.hdf5"

    index_filename = None
    build_vocabulary(corpus_file,index_filename, max_sentences)
