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
from spacy.vocab import Vocab


def load_dataset(hdf5_file):
    # print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "sentences_lemmatised"
    dataset = hdf5_file[dataset_name]
    return dataset


def load_spacy():
    t0 = time.time()
    # load tokenizer only
    nlp = English(entity=False, load_vectors=False, parser=False, tagger=False, matcher=False)
    nlp.vocab.strings._map =
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


def build_vocabulary(corpus_file, output_file=None, max_sentences=0):
    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset = load_dataset(input_hdf5)

    if max_sentences > 0:
        num_sentences = min(max_sentences, len(dataset))
    else:
        num_sentences = len(dataset)

    nlp = load_spacy()
    tokenizer = nlp.tokenizer

    freq = Counter()

    # ************************************ PROCESS VOCABULARY **************************************************
    # load one sentence in memory at a time
    #perhaps this is too slow, should I load chunks at a time and then create a generator for each slice inthe chunk?
    sentence_gen = (dataset[i] for i in range(num_sentences))
    for sentence in tqdm(tokenizer.pipe(sentence_gen, n_threads=4, batch_size=4000), total=num_sentences):
        tokens = [replace_w_token(token, nlp) for token in sentence if is_valid_token(token)]

        for token in tokens:
            freq[token.orth] += 1

    input_hdf5.close()
    tqdm.write("{0} unique words".format(len(freq)))
    # order by frequency
    freq = freq.most_common()

    for i in range(10):
        print("{0}:{1}".format(nlp.vocab.strings[freq[i][0]], freq[i][1]))

    # ************************************ WRITE VOCABULARY TO HDF5 **********************************************
    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'w')
        word_ids = range(len(freq))
        # convoluted scheme of spaCy
        vocabulary = np.array([nlp.vocab.strings[lex_id] for (lex_id, _) in freq])

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
    max_sentences = 0

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = "/data/datasets/wacky.hdf5"
    corpus_file = home + corpus_file
    output_file = home + "/data/results/wacky_index.hdf5"

    build_vocabulary(corpus_file, output_file, max_sentences)
