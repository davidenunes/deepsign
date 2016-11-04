#!/usr/bin/env python

import time
import h5py
import os.path
from spacy.en import English
import numpy as np
from tqdm import tqdm
from collections import Counter

from deepsign.nlp.tokenization import Tokenizer
from deepsign.nlp import is_token as itk
from deepsign.nlp import stoplist
from deepsign.utils.views import divide_slice
from itertools import chain


def load_tokenizer():
    tokenizer = Tokenizer()
    return tokenizer


def load_dataset(hdf5_file):
    # print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "sentences_lemmatised"
    dataset = hdf5_file[dataset_name]
    return dataset


# TODO check what other useless tokens are put in wacky
def valid_token(token):
    if itk.is_punct(token) or itk.is_space(token) or itk.is_copyright(token):
        return False

    # TODO tokenizer should support custom tokens often these might be odd patterns
    # that get torn appart by the tokenizer like @ord@ and @card@ because @ is not
    # something used as a separator
    custom_stop = ("@card", "@ord")
    if token in custom_stop:
        return False

    return True


def replace_token(token):
    result = token
    if itk.is_url(token):
        result = "T_URL"
    elif itk.is_email(token):
        result = "T_EMAIL"
    elif itk.is_currency(token):
        result = "T_CURRENCY"

    return result


def build_vocabulary(corpus_file, output_file=None, max_sentences=0):
    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset = load_dataset(input_hdf5)

    if max_sentences > 0:
        num_sentences = min(max_sentences, len(dataset))
    else:
        num_sentences = len(dataset)

    tokenizer = load_tokenizer()

    freq = Counter()

    # ************************************ PROCESS VOCABULARY **************************************************
    # load one sentence in memory at a time
    # perhaps this is too slow, should I load chunks at a time and then create a generator for each slice in the chunk?
    chunk_size = 2
    n_chunks = num_sentences // chunk_size
    chunk_slices = divide_slice(num_sentences, n_chunks)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    def chunk_it(c):
        for i in range(len(c)):
            yield c[i]

    sentence_gen = chain.from_iterable(chunk_it(c) for c in chunk_gen)

    for sentence in tqdm(sentence_gen, total=num_sentences):
        tokens = tokenizer.tokenize(sentence)
        tokens = [replace_token(token) for token in tokens if valid_token(token)]
        # filter stopwords
        #tokens = [token for token in tokens if not itk.is_stopword(token)]

        #tqdm.write(str(tokens))
        for token in tokens:
            freq[token] += 1

    input_hdf5.close()
    tqdm.write("{0} unique words".format(len(freq)))
    # order by frequency
    freq = freq.most_common()

    for i in range(10):
        (w,f) = freq[i]
        print("{0}:{1}".format(w, f))

    # ************************************ WRITE VOCABULARY TO HDF5 **********************************************
    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'w')
        word_ids = range(len(freq))

        # encode explicitly so that hdf5 can take an array of variable length strings and store it
        vocabulary = np.array([freq[i][0].encode("utf8") for i in range(len(freq))])


        # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
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
    max_sentences = 1000000

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = home + "/data/datasets/wacky.hdf5"
    index_filename = home + "/data/results/wacky_vocabulary_stop.hdf5"

    #index_filename = None
    build_vocabulary(corpus_file,index_filename, max_sentences)
