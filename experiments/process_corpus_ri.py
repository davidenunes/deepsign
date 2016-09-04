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
from deepsign.io.h5utils import batch_write
import sys

def load_dataset(hdf5_file):
    print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "ukwac_sentences"
    dataset = hdf5_file[dataset_name]
    return dataset

def load_spacy():
    t0 = time.time()
    # load tokenizer only
    nlp = English(entity=False, load_vectors=False, parser=True, tagger=True)
    t1 = time.time()
    print("Done: {0:.2f} secs ".format(t1 - t0))
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
def replace_w_token(t):
    result = t.orth_

    if t.like_url:
        result = "T_URL"
    elif t.like_email:
        result = "T_EMAIL"

    return result
    # work on a dataset row slice


def synch_occurr(occurr_dict, occurr_dataset):
    word_ids = sorted(occurr_dict.keys())
    max_id = word_ids[-1]
    if max_id >= len(occurr_dataset):
        add_rows = max_id-len(occurr_dataset) + 1
        occurr_dataset.resize(occurr_dataset.shape[0] + add_rows, 0)

    # update dataset file
    for i in word_ids:
        occurr_dataset[i] += occurr_dict[i].to_vector()


def process_corpus(corpus_file, result_file, max_sentences=0, window_size=3, ri_dim=1000, ri_active=10):

    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset = load_dataset(input_hdf5)

    if max_sentences > 0:
        num_sentences = min(max_sentences,len(dataset))
    else:
        num_sentences = len(dataset)

    output_hdf5 = h5py.File(result_file, 'a')

    # ****************************************** PROCESS CORPUS ********************************************************
    # TODO turn this into a general purpose module
    ri_gen = RandomIndexGenerator(dim=ri_dim, active=ri_active)
    sign_index = SignIndex(ri_gen)

    # avg are stored as is since there is no guarantee that these will be sparse (depends on the params)
    occurr_dataset = output_hdf5.create_dataset("ri_sum", shape=(0, ri_dim), maxshape=(None, ri_dim), compression="gzip")
    occurr = dict()
    occurr_synch_t = 500000


    frequencies = dict()
    data_gen = (dataset[i][0] for i in range(num_sentences))

    nlp = load_spacy()

    num_updates = 0
    for sentence in tqdm(nlp.pipe(data_gen, n_threads=8, batch_size=2000), total=num_sentences):
        # TODO all caps to lower except entities
        tokens = [replace_w_token(token) for token in sentence if is_valid_token(token)]

        sign_index.add_all(tokens)

        # get sliding windows of given size
        s_windows = sliding(tokens, window_size=window_size)


        # Encode each window as a bag-of-words and add to occurrences
        for window in s_windows:
            bow_vector = to_bow(window, sign_index)
            bow_vector = views.np_to_sparse(bow_vector)
            sign_id = sign_index.get_id(window.target)

            if sign_id not in occurr:
                occurr[sign_id] = bow_vector
                frequencies[sign_id] = 1
            else:
                current_vector = occurr[sign_id]
                occurr[sign_id] = bow_vector + current_vector
                frequencies[sign_id] += 1
            num_updates += 1

        if num_updates >= occurr_synch_t:
            tqdm.write("Synching occurrences...")
            synch_occurr(occurr, occurr_dataset)
            occurr = dict()
            num_updates = 0
            tqdm.write("done")

    synch_occurr(occurr, occurr_dataset)

    # ************************************** END PROCESS CORPUS ********************************************************
    input_hdf5.close()

    # ************************************** WRITE INDEX & FREQ ********************************************************
    word_ids = range(len(occurr_dataset))
    print("processing {0} word vectors".format(len(word_ids)))

    print("writing to ", result_file)

    dataset_name = "{0}_{1}".format(ri_dim, ri_active)
    print("dataset: " + dataset_name)

    vocabulary = np.array([sign_index.get_sign(w_id).encode("UTF-8") for w_id in word_ids])

    # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
    dt = h5py.special_dtype(vlen=str)
    vocabulary_data = output_hdf5.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="gzip")
    print("vocabulary data written")

    frequencies = np.array([frequencies[w_id] for w_id in word_ids])
    frequency_data = output_hdf5.create_dataset("frequencies", data=frequencies, compression="gzip")
    print("count data written")

    # random indexing vectors are stored in sparse mode (active indexes only),
    # reconstruct using ri.from_sparse(dim,active,active_list) to get a RandomIndex object
    ri_vectors = [sign_index.get_ri(w.decode("UTF-8")) for w in vocabulary]
    # store random indexes as a list of positive indexes followed by negative indexes
    ri_sparse = np.array([ri.positive + ri.negative for ri in ri_vectors])
    index_data = output_hdf5.create_dataset(dataset_name, data=ri_sparse, compression="gzip")
    print("random index vectors written")

    index_data.attrs["dimension"] = ri_dim
    index_data.attrs["active"] = ri_active

    output_hdf5.close()

if __name__ == '__main__':
    # model parameters
    window_size = 3
    max_sentences = 1000000
    ri_dim = 1000
    ri_active = 10

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = "/data/datasets/wacky.hdf5"
    corpus_file = home + corpus_file
    result_file = home + "/data/results/random_indexing_1M.hdf5"

    process_corpus(corpus_file, result_file, max_sentences, window_size, ri_dim, ri_active)
