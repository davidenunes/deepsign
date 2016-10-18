#!/usr/bin/env python

# from nltk.tokenize import StanfordTokenizer
# tokenizer = StanfordTokenizer(path_to_jar="/home/davex32/dev/deepsign/libs/stanford-postagger/stanford-postagger.jar")
# tokenizer.tokenize("something something")

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
    if token in stoplist.ENGLISH or \
            itk.is_punct(token) or \
            itk.is_space(token):

        return False

    custom_stop = ("'s", "@card", "@ord")
    if token in custom_stop:
        return False

    return True


# TODO replace numbers with T_NUMBER ?
# TODO replace time with T_TIME?
special_tokens = {"URL": "T_URL", "EMAIL": "T_EMAIL"}


def replace_token(token):
    result = token
    if itk.is_url(token):
        result = special_tokens["URL"]
    elif itk.is_email(token):
        result = special_tokens["EMAIL"]

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
    # perhaps this is too slow, should I load chunks at a time and then create a generator for each slice inthe chunk?
    chunk_size = 2
    n_chunks = num_sentences // chunk_size
    chunk_slices = divide_slice(num_sentences, n_chunks)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    def chunk_it(c):
        for i in range(len(c)):
            yield c[i]

    sentence_gen = chain.from_iterable(chunk_it(c) for c in chunk_gen)

    for sentence in tqdm(sentence_gen, total=num_sentences):
        print(sentence)
        tokens = tokenizer.tokenize(sentence)
        tokens = [replace_token(token) for token in tokens if valid_token(token)]

        #tqdm.write(str(tokens))

        for token in tokens:
            freq[token] += 1

    input_hdf5.close()
    tqdm.write("{0} unique words".format(len(freq)))
    # order by frequency
    # freq = freq.most_common()

    # for i in range(10):
    #    print("{0}:{1}".format(freq[i][0], freq[i][1]))

    # ************************************ WRITE VOCABULARY TO HDF5 **********************************************
    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'w')
        word_ids = range(len(freq))
        # convoluted scheme of spaCy
        vocabulary = np.array([w for (w, _) in freq])

        # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
        dt = h5py.special_dtype(vlen=str)
        output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
        print("vocabulary written")

        freq = np.array([f for (_, f) in freq])
        output_hdf5.create_dataset("frequencies", data=freq, compression="gzip")
        print("frequencies written")
        output_hdf5.close()
        print("done")


if __name__ == '__main__':
    # model parameters
    max_sentences = 10000

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = "/data/datasets/wacky.hdf5"
    corpus_file = home + corpus_file
    output_file = home + "/data/results/wacky_index.hdf5"

    build_vocabulary(corpus_file, None, max_sentences)
