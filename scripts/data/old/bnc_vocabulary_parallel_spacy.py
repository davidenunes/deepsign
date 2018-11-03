#!/usr/bin/env python

import argparse
import os.path
from collections import Counter
from multiprocessing import Pool

import h5py
import numpy as np
from tqdm import tqdm

from deepsign.data.corpora.pipe import BNCPipe
from deepsign.data.views import subset_chunk_it, divide_slices


def word_frequencies(args):
    """
    :param fname: name of the hdf5 file containing the corpus
    :param data_slice: a range with the subset of the file to be read
    :return: a Counter with the frequency of the tokens found
    """
    fname, data_slice, lemmatize = args
    input_hdf5 = h5py.File(fname, 'r')
    dataset_name = "sentences"
    dataset = input_hdf5[dataset_name]
    gen = subset_chunk_it(dataset, data_slice, chunk_size=100)

    pbar = tqdm(total=len(data_slice))

    pipe = BNCPipe(gen,lemmas=lemmatize)
    freq = Counter()

    for tokens in pipe:
        #print(tokens)
        for token in tokens:
            normal_token = token.lower()
            freq[normal_token] += 1
        pbar.update(1)

    input_hdf5.close()
    return freq


def parallel_word_count(corpus_file, output_file,lemmatize, max_rows=None, n_processes=8):
    # get data slices
    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset_name = "sentences"
    dataset = input_hdf5[dataset_name]
    nrows = len(dataset)
    input_hdf5.close()

    if max_rows !=-1 and max_rows < nrows:
        nrows = max_rows

    data_slices = divide_slices(nrows, n_processes, 0)

    args = [(corpus_file, data_slice, lemmatize) for data_slice in data_slices]
    pool = Pool(n_processes)

    result = pool.map(func=word_frequencies, iterable=args)
    pool.close()

    # agglomerate results
    freq = Counter()
    for freq_i in result:
        freq = freq + freq_i
    freq = freq.most_common()
    print("{0} unique words".format(len(freq)))
    for i in range(10):
        (w, f) = freq[i]
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
    parser = argparse.ArgumentParser(description="Neural Random Projections arguments")
    parser.add_argument('-lemmas', dest="lemmas", type=bool, default=True)
    parser.add_argument('-data_dir', dest="data_dir", type=str, default="/data/gold_standards/")
    parser.add_argument('-corpus_file', dest="corpus_file", type=str, default="bnc.hdf5")
    parser.add_argument('-out_file', dest="out_file", type=str, default="bnc_vocab_lemma.hdf5")
    parser.add_argument('-n_sentences', dest="n_sentences", type=int, default=-1)
    args = parser.parse_args()

    # all sentences
    max_sentences = None

    # corpus and output assets
    home = os.getenv("HOME")
    corpus_file = home + args.data_dir + args.corpus_file
    index_filename = home + args.data_dir + args.out_file

    #index_filename = None
    parallel_word_count(corpus_file, index_filename, args.lemmas, args.n_sentences,n_processes=2)
