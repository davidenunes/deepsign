#!/usr/bin/env python

import h5py
import os.path

from deepsign.utils.views import sliding_windows as sliding
from deepsign.utils.views import chunk_it
from deepsign.nlp.tokenization import Tokenizer

from experiments.pipe.wacky_pipe import WaCKyPipe

from deepsign.rp.index import SignIndex
from deepsign.rp.ri import RandomIndexGenerator
from deepsign.rp.encode import to_bow
import deepsign.utils.views as views
import numpy as np
from tqdm import tqdm


def synch_occurr(occurr_dict, occurr_dataset):
    word_ids = sorted(occurr_dict.keys())
    max_id = word_ids[-1]
    if max_id >= len(occurr_dataset):
        add_rows = max_id-len(occurr_dataset) + 1
        occurr_dataset.resize(occurr_dataset.shape[0] + add_rows, 0)

    # update dataset file
    for i in word_ids:
        occurr_dataset[i] += occurr_dict[i].to_vector()


def process_corpus(input_file, output_file, max_rows=0, window_size=3, ri_dim=1000, ri_active=10):
    input_hdf5 = h5py.File(input_file, 'r')
    dataset_name = "sentences_lemmatised"
    dataset = input_hdf5[dataset_name]

    if max_rows > 0:
        nrows = min(max_rows, len(dataset))
    else:
        nrows = len(dataset)

    gen = chunk_it(dataset, nrows, chunk_size=200)
    tokenizer = Tokenizer()
    pipe = WaCKyPipe(gen, tokenizer, filter_stop=False)

    ri_gen = RandomIndexGenerator(dim=ri_dim, active=ri_active)
    sign_index = SignIndex(ri_gen)

    occurr = dict()

    # avg are stored as is since there is no guarantee that these will be sparse (depends on the params)
    if output_file is not None:
        output_hdf5 = h5py.File(output_file, 'a')
        occurr_dataset = output_hdf5.create_dataset("ri_sum", shape=(0, ri_dim), maxshape=(None, ri_dim), compression="gzip")
        occurr_synch_t = 1000

    num_updates = 0
    for tokens in tqdm(pipe, total=nrows):
        sign_index.add_all(tokens)

        # get sliding windows of given size
        s_windows = sliding(tokens, window_size=window_size)

        # encode each window as a bag-of-words and add to occurrencies
        for window in s_windows:
            bow_vector = to_bow(window, sign_index)
            bow_vector = views.np_to_sparse(bow_vector)
            sign_id = sign_index.get_id(window.target)

            if sign_id not in occurr:
                occurr[sign_id] = bow_vector
            else:
                current_vector = occurr[sign_id]
                occurr[sign_id] = bow_vector + current_vector
            num_updates += 1


        #if num_updates >= occurr_synch_t:
        #    tqdm.write("Synching occurrences...")
        #    synch_occurr(occurr, occurr_dataset)
        #    occurr = dict()
        #    num_updates = 0
        #    tqdm.write("done")

    if output_file is not None:
        synch_occurr(occurr, occurr_dataset)

    # ************************************** END PROCESS CORPUS ********************************************************

    if output_file is not None:
        word_ids = range(len(occurr_dataset))
        print("processing {0} word vectors".format(len(word_ids)))

        print("writing to ", output_file)

        dataset_name = "{0}_{1}".format(ri_dim, ri_active)
        print("dataset: " + dataset_name)

        # random indexing vectors are stored in sparse mode (active indexes only),
        # reconstruct using ri.from_sparse(dim,active,active_list) to get a RandomIndex object
        ri_vectors = [sign_index.get_ri(w.decode("UTF-8")) for w in word_ids]
        # store random indexes as a list of positive indexes followed by negative indexes
        ri_sparse = np.array([ri.positive + ri.negative for ri in ri_vectors])
        index_data = output_hdf5.create_dataset(dataset_name, data=ri_sparse, compression="gzip")
        print("random index vectors written")

        index_data.attrs["dimension"] = ri_dim
        index_data.attrs["active"] = ri_active

        output_hdf5.close()

if __name__ == '__main__':
    # model parameters
    max_sentences = 10

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = home + "/data/datasets/wacky.hdf5"
    results_file = home + "/data/results/ri.hdf5"

    results_file = None
    process_corpus(corpus_file, results_file, max_sentences,
                   window_size=3, ri_dim=1000, ri_active=10)
