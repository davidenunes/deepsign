#!/usr/bin/env python

import time
import h5py
import os.path
from spacy.en import English
from deepsign.utils.views import sliding_windows as sliding
from deepsign.rp.index import SignIndex
from deepsign.rp.ri import RandomIndexGenerator
from deepsign.rp.encode import to_bow
import deepsign.utils.views as views
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from deepsign.utils.profiling import total_size
from collections import Counter
from deepsign.utils.views import divide_slice


def load_dataset(hdf5_file):
    # print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "sentences_lemmatised"
    dataset = hdf5_file[dataset_name]
    return dataset


def load_spacy():
    t0 = time.time()
    # load tokenizer only
    nlp = English(entity=False, load_vectors=False, parser=False, tagger=False)
    t1 = time.time()
    # print("Done: {0:.2f} secs ".format(t1 - t0))
    return nlp


# TODO check what other useless tokens are put in wacky
def is_valid_token(token):
    if token.is_punct or token.is_stop:
        return False

    w = token.orth_
    custom_stop = ["'s", "@card@", "@ord@"]
    for stop in custom_stop:
        if w == stop:
            return False

    return True


# TODO replace numbers with T_NUMBER ?
# TODO replace time with T_TIME?
special_tokens = {"URL": "T_URL", "EMAIL": "T_EMAIL"}


def replace_w_token(t, nlp):
    result = t.orth_

    if t.like_url:
        result = special_tokens["URL"]
    elif t.like_email:
        result = special_tokens["EMAIL"]

    lex = nlp.vocab[result]
    return lex
    # work on a dataset row slice


def synch_occurr(occurr_dict, occurr_dataset):
    word_ids = sorted(occurr_dict.keys())
    max_id = word_ids[-1]
    if max_id >= len(occurr_dataset):
        add_rows = max_id - len(occurr_dataset) + 1
        occurr_dataset.resize(occurr_dataset.shape[0] + add_rows, 0)

    # update dataset file
    for i in word_ids:
        occurr_dataset[i] += occurr_dict[i].to_vector()


def update_index(tokens, sign_index, freq):
    sign_index.add_all(tokens)

    for token in tokens:
        tk_id = sign_index.get_id(token)
        freq[tk_id] += 1


def unique_words(dataset_file, slice_range):
    print("worker started on {0}".format(slice_range))

    h5f = h5py.File(dataset_file, 'r')
    dataset = load_dataset(h5f)

    freq = Counter()

    # load one sentence in memory at a time
    sentence_gen = (dataset[i] for i in slice_range)
    for sentence in tqdm(nlp.pipe(sentence_gen, n_threads=8, batch_size=2000), total=len(slice_range)):
        tokens = [replace_w_token(token, nlp) for token in sentence if is_valid_token(token)]

        for token in tokens:
            freq[token.orth] += 1

    h5f.close()

    return freq


def process_corpus_parallel(corpus_file, output_file=None, max_sentences=0):
    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset = load_dataset(input_hdf5)

    if max_sentences > 0:
        num_sentences = min(max_sentences, len(dataset))
    else:
        num_sentences = len(dataset)

    input_hdf5.close()
    n_workers = 1  # mp.cpu_count()
    print("starting {0} workers".format(n_workers))

    pool = mp.Pool(n_workers)
    dataset_slices = divide_slice(num_sentences, n_workers)

    worker_fn = partial(unique_words, corpus_file)
    word_count_list = pool.map(func=worker_fn, iterable=dataset_slices)

    pool.close()
    pool.join()

    # merge frequencies
    word_counts = sum(word_count_list, Counter())
    word_counts = word_counts.most_common()

    for i in range(2000):
        (lex_id, f) = word_counts[i]
        try:
            print((nlp.vocab[lex_id].orth_, f))
        except Exception:
            print("ERROR WITH:")
            print(lex_id)
            print(nlp.vocab[lex_id])

    print("{0} unique words".format(len(word_counts)))

    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'w')

        word_ids = range(len(word_counts))

        # convoluted scheme of spaCy
        vocabulary = np.array([nlp.vocab.strings[lex_id] for (lex_id, _) in word_counts])

        # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
        dt = h5py.special_dtype(vlen=str)
        vocabulary_data = output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
        print("vocabulary written")

        freq = np.array([f for (_, f) in word_counts])
        frequency_data = output_hdf5.create_dataset("frequencies", data=freq, compression="gzip")
        print("frequencies written")
        output_hdf5.close()
        print("done")


print("loading spaCy")
nlp = load_spacy()
# load special tokens to vocabulary
# found some problems with SPACY, if they try to load vocabulary from different threads it's all fucked up!!!! so better
# load special tokens first
for st in special_tokens:
    nlp.vocab[special_tokens[st]]
print("spaCy loaded")

if __name__ == '__main__':
    # model parameters
    max_sentences = 20000

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = "/data/datasets/wacky.hdf5"
    corpus_file = home + corpus_file
    output_file = home + "/data/results/wacky_index.hdf5"

    process_corpus_parallel(corpus_file, None, max_sentences)
