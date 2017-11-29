import numpy as np
from itertools import chain, tee
from deepsign.rp.ri import RandomIndex


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def divide_slice(n, n_slices=1, offset=0):
    """ Splits a vector with ngram_size elements equally into n_slices
    returning a list of index ranges for that vector, each range corresponds
    to a slice.

    :param n: number of elements in the vector
    :param n_slices: number of slices the vector is to be split into
    :return: a list of slices for the vector
    """
    len_split = int(n / n_slices)
    num_indexes = n_slices - 1

    ss = [0]
    for s in range(len_split, len_split * n_slices, len_split):
        ss.append(s)

    ss.append(n)
    ranges = [range(s[0] + offset, s[1] + offset) for s in _pairwise(ss)]

    return ranges


class Window:
    """ A window contains:
        a left []
        a target which is in the center of the window
        a right []
    """

    def __init__(self, left, target, right):
        self.left = left
        self.target = target
        self.right = right

    def __str__(self):
        return "(" + str(self.left) + "," + self.target + "," + str(self.right) + ")"


def sliding_windows(seq, window_size=1):
    """ converts a sequence of strings to a sequence of windows

    :param seq: a sequence to be sliced into windows
    :param window_size: the size of the window around each element
    :return: an array of Window instances
    """
    elem_indexes = range(0, len(seq))
    n_elems = len(seq)

    windows = []
    # create a sliding window for each elem
    for w in elem_indexes:
        # lower limits
        wl = max(0, w - window_size)
        wcl = w

        # upper limits
        wch = n_elems if w == n_elems - 1 else min(w + 1, n_elems - 1)
        wh = w + min(window_size + 1, n_elems)

        # create window
        left = seq[wl:wcl]
        target = seq[w]
        right = seq[wch:wh]

        windows.append(Window(left, target, right))

    return windows


class SparseArray(object):
    def __init__(self, dim, active, values):
        self.dim = dim
        self.active = active
        self.values = values

    def to_vector(self):
        v = np.zeros(self.dim)
        v[self.active] = self.values
        return v

    def __add__(self, other):
        s_v = self.to_vector()
        o_v = other.to_vector()

        r = s_v + o_v
        active = np.nonzero(r)

        return SparseArray(self.dim, active, r[active])


def np_to_sparse(sparse_array):
    """ Converts a 1D numpy array to a sparse version
    :param sparse_array: the array to be converted
    :type sparse_array np.ndarray
    """
    active = np.nonzero(sparse_array)
    values = sparse_array[active]
    dim = sparse_array.shape[0]
    result = SparseArray(dim=dim, active=active, values=values)
    return result


def ri_to_sparse(random_index):
    """Converts a random index to a SparseArray object"""
    active = random_index.positive + random_index.negative
    values = [1] * len(random_index.positive) + [-1] * len(random_index.negative)

    return SparseArray(dim=random_index.dim, active=active, values=values)


def chunk_it(dataset, n_rows=None, chunk_size=1):
    """
    Allos to iterate over dataset by loading chunks at a time using slices
    up until a given nrows

    :param dataset: the dataset we wish to iterate over
    :param n_rows: number of rows we want to take from the dataset (start at 0)
    :param chunk_size: the chunk size to be loaded into the memory
    :return: and iterator over the elements of dataset with buffered slicing
    """
    if n_rows is None:
        n_rows = len(dataset)

    if chunk_size > n_rows:
        chunk_size = n_rows

    n_chunks = n_rows // chunk_size
    chunk_slices = divide_slice(n_rows, n_chunks)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    row_gen = chain.from_iterable((c[i] for i in range(len(c))) for c in chunk_gen)
    return row_gen


def subset_chunk_it(dataset, data_range, chunk_size=1):
    """Allows to iterate over a given subset of a given dataset by loading chunks at a time

    dataset: the given dataset
    data_range: a range from which we will extract the ngram_size chunks to be loaded from the dataset
    chunk_size: length of each chunk to be loaded, this determines the number of chunks

    """
    nrows = len(data_range)

    if chunk_size > nrows:
        chunk_size = nrows

    n_chunks = nrows // chunk_size
    chunk_slices = divide_slice(nrows, n_chunks, data_range.start)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    row_gen = chain.from_iterable((c[i] for i in range(len(c))) for c in chunk_gen)
    return row_gen



def ngram_windows(seq, window_size=1):
    """ converts a list of strings to a list of lists of strings each with
    a given window size.

    :param seq: list of strings
    :param window_size: size for the ngram windows
    :return:
    """
    grams = [seq[i:i + window_size] for i in range(len(seq) - window_size + 1)]
    result = [ngram for ngram in grams]
    return result