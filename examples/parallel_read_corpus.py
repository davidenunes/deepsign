#!/usr/bin/env python
# Example for the use of multiprocessing and spaCy to read and process a corpus dataset
# in parallel. The problem with this is that in order to transform it into a parallel processing
# with random indexing, the sign index update becomes the bottleneck
# perhaps this bottleneck is not as sever as I think it is, I should just implement it and profile the code
# in production, some kind of system providing the sign index would have to exist anyway unless I find a way
# to learn with run using different vector basis for each word.

import time
import h5py
import os.path
from spacy.en import English

from deepsign.data.views import divide_slice
from deepsign.rp.index import SignIndex
from deepsign.rp.ri import Generator


from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

# *************************
# model parameters
# *************************
window_size = 3
ri_dim = 600
ri_num_active = 4


def load_dataset():
    home = os.getenv("HOME")
    corpus_file = "/data/gold_standards/wacky.hdf5"
    result_path = home + "/data/results/"
    corpus_file = home + corpus_file

    print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "sentences_lemmatised"

    # open hdf5 file and get the dataset
    h5f = h5py.File(corpus_file, 'r')
    dataset = h5f[dataset_name]
    return dataset

# do something with the dataset

# Create Sign RI Index
ri_gen = Generator(dim=ri_dim, active=ri_num_active)
sign_index = SignIndex(ri_gen)

max_sentences = 200000


def load_spacy():
    t0 = time.time()
    # load tokenizer only
    nlp = English(entity=False, load_vectors=False, parser=True, tagger=True)
    t1 = time.time()
    print("Done: {0:.2f} secs ".format(t1 - t0))
    return nlp

nlp = load_spacy()


def is_invalid_token(token):
    if token.is_punct or token.is_stop:
        return True

    w = token.orth_

    # some words are tokenised with 's and ngram_size't, apply this before filtering stop words
    custom_stop = ["'s",
                   "@card@",
                   "@ord@",
                   ]

    for stop in custom_stop:
        if w == stop:
            return True

    return False


def replace_w_token(t):
    result = t.orth_

    if t.like_url:
        result = "T_URL"
    elif t.like_email:
        result = "T_EMAIL"

    return result
    # work on a dataset row slice


def do_work(slice_range):
    print("worker {0} is starting".format(mp.current_process()))

    dataset = load_dataset()

    # load one sentence in memory at a time
    sentence_gen = (dataset[i][0] for i in slice_range)

    # since pipe work based on a generator (sentence_it) we have to provide the max number of iterations
    for sentence in tqdm(nlp.pipe(sentence_gen, n_threads=8, batch_size=500), total=len(slice_range)):
        tokens = [replace_w_token(tk) for tk in sentence if not is_invalid_token(tk)]


def parallel_process_corpus():
    n_workers = 4
    pool = Pool(n_workers)
    dataset_slices = divide_slice(max_sentences, n_workers)
    t0 = time.time()
    pool.map(func=do_work, iterable=dataset_slices)
    pool.close()
    pool.join()
    t1 = time.time()
    time.sleep(1)
    print("Done: {0:.2f} secs ".format(t1 - t0))

if __name__ == '__main__':
    parallel_process_corpus()
