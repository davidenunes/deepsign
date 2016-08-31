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
from deepsign.io.h5store import batch_write

def load_dataset(hdf5_file):
    print("Reading hdf5 dataset from: ", corpus_file)
    dataset_name = "ukwac_sentences"
    dataset = hdf5_file[dataset_name]
    return dataset


def write_results(result_file, sign_index, frequencies, occurrences):
    h5f = h5py.File(result_file, 'w')

    word_ids = occurrences.keys()
    print("processing {0} word vectors".format(len(word_ids)))

    print("writing to ", result_file)
    dim = sign_index.feature_dim()
    act = sign_index.feature_act()

    dataset_name = "{0}_{1}".format(dim, act)
    print("dataset: " + dataset_name)

    vocabulary = np.array([sign_index.get_sign(w_id).encode("UTF-8") for w_id in word_ids])

    # the hdf5 needs to store variable-length strings with a specific encoding (UTF-8 in this case)
    dt = h5py.special_dtype(vlen=str)
    vocabulary_data = h5f.create_dataset("vocabulary", data=vocabulary, dtype=dt, compression="lzf")
    print("vocabulary data written")

    frequencies = np.array([frequencies[w_id] for w_id in word_ids])
    frequency_data = h5f.create_dataset("frequencies", data=frequencies, compression="lzf")
    print("count data written")

    # random indexing vectors are stored in sparse mode (active indexes only),
    # reconstruct using ri.from_sparse(dim,active,active_list) to get a RandomIndex object
    ri_vectors = [sign_index.get_ri(w.decode("UTF-8")) for w in vocabulary]
    # store random indexes as a list of positive indexes followed by negative indexes
    ri_sparse = np.array([ri.positive + ri.negative for ri in ri_vectors])
    index_data = h5f.create_dataset(dataset_name, data=ri_sparse, compression="lzf")
    print("random index vectors written")

    index_data.attrs["dimension"] = dim
    index_data.attrs["active"] = act


    # avg are stored as is since there is no guarantee that these will be sparse (depends on the params)
    sum_vectors = h5f.create_dataset(dataset_name + "_sum", shape=(len(word_ids),dim), compression="lzf")

    # num batches to store
    num_words = len(word_ids)
    occurrence_gen = (occurrences[w_id].to_vector() for w_id in range(num_words))

    batch_write(sum_vectors,occurrence_gen,num_words,1000,progress=True)
    time.sleep(0.5)
    print("random index sum vectors written")

    h5f.close()


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


def process_corpus(corpus_file, result_file, max_sentences=0, window_size=3, ri_dim=1000, ri_active=10):

    h5f = h5py.File(corpus_file, 'r')
    dataset = load_dataset(h5f)

    if max_sentences > 0:
        num_sentences = min(max_sentences,len(dataset))
    else:
        num_sentences = len(dataset)

    # ****************************************** PROCESS CORPUS ********************************************************
    # TODO turn this into a general purpose module
    ri_gen = RandomIndexGenerator(dim=ri_dim, active=ri_active)
    sign_index = SignIndex(ri_gen)
    occurrences = dict()
    frequencies = dict()
    data_gen = (dataset[i][0] for i in range(num_sentences))

    nlp = load_spacy()
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

            if sign_id not in occurrences:
                occurrences[sign_id] = bow_vector
                frequencies[sign_id] = 1
            else:
                current_vector = occurrences[sign_id]
                occurrences[sign_id] = bow_vector + current_vector
                frequencies[sign_id] += 1

    # ************************************** END PROCESS CORPUS ********************************************************
    h5f.close()

    write_results(result_file,sign_index,frequencies,occurrences)

if __name__ == '__main__':
    # model parameters
    window_size = 3
    max_sentences = 2030
    ri_dim = 600
    ri_active = 4

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = "/data/datasets/wacky.hdf5"
    corpus_file = home + corpus_file
    result_file = home + "/data/results/random_indexing_test.hdf5"

    process_corpus(corpus_file, result_file, max_sentences, window_size, ri_dim, ri_active)
