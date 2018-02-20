import os
from multiprocessing import Pool

import h5py
from tqdm import tqdm

from deepsign.data.corpora.wacky_pipe import WaCKyPipe
from deepsign.nlp.tokenization import Tokenizer
from deepsign.rp.encode import to_bow
from deepsign.rp.index import TrieSignIndex
from deepsign.rp.ri import Generator
from deepsign.data.views import windows, np_to_sparse, divide_slice
from deepsign.data.views import subset_chunk_it

# global for sign index
sign_index = None


def init_lock(l):
    global lock
    lock = l


def text_to_ri(args):
    (fname, data_slice, window_size) = args

    input_hdf5 = h5py.File(fname, 'r')
    dataset_name = "sentences"
    dataset = input_hdf5[dataset_name]
    gen = subset_chunk_it(dataset, data_slice, chunk_size=250)

    pbar = tqdm(total=len(data_slice))

    tokenizer = Tokenizer()
    pipe = WaCKyPipe(gen, tokenizer, filter_stop=False)

    global sign_index
    ri_vectors = dict()

    for tokens in pipe:
        # get sliding windows of given size
        s_windows = windows(tokens, window_size)

        # encode each window as a bag-of-words and add to occurrencies
        for window in s_windows:
            # pbar.write(str(window))
            # lock.acquire()
            bow_vector = to_bow(window, sign_index)
            # lock.release()
            bow_vector = np_to_sparse(bow_vector)
            sign_id = sign_index.get_id(window.target)

            if sign_id not in ri_vectors:
                ri_vectors[sign_id] = bow_vector
            else:
                current_vector = ri_vectors[sign_id]
                ri_vectors[sign_id] = bow_vector + current_vector

        pbar.update(1)

    return ri_vectors


def parallel_ri(corpus_file, max_rows=None, window_size=3, n_processes=8):
    # get data slices
    input_hdf5 = h5py.File(corpus_file, 'r')
    dataset_name = "sentences"
    dataset = input_hdf5[dataset_name]
    nrows = len(dataset)
    input_hdf5.close()

    if max_rows is not None and max_rows < nrows:
        nrows = max_rows

    data_slices = divide_slice(max_rows, n_processes, 0)

    args = [(corpus_file, data_slice, window_size) for data_slice in data_slices]

    # share a global lock to avoid messing sign index on lookups
    # l = Lock()
    # pool = Pool(initializer=init_lock(l), initargs=(l,), processes=n_processes)
    pool = Pool(processes=n_processes)

    result = pool.map(func=text_to_ri, iterable=args)
    pool.close()


if __name__ == '__main__':
    # model parameters
    max_sentences = 10000

    # corpus and output files
    home = os.getenv("HOME")
    corpus_file = home + "/data/gold_standards/wacky_1M.hdf5"
    output_vectors = home + "/data/results/wacky_ri_1M.hdf5"

    # load sign index
    print("loading vocabulary")
    vocab_file = home + "/data/results/wacky_vocab_1M.hdf5"
    h5v = h5py.File(vocab_file, 'r')
    vocabulary = h5v["vocabulary"]
    frequencies = h5v["frequencies"]

    ri_gen = Generator(dim=1000, num_active=10)
    sign_index = TrieSignIndex(generator=ri_gen,
                               vocabulary=list(vocabulary[()]),
                               pregen_indexes=True)

    print("done")

    # index_filename = None
    parallel_ri(corpus_file, max_sentences, window_size=3, n_processes=16)
