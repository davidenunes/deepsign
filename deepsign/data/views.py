import numpy as np
import itertools
from functools import partial


def _pairwise(iterable):
    """
    Example:
        s -> (s0,s1), (s1,s2), (s2, s3), ...

    Returns:
        an iterable of tuples with each element paired with the next.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def divide_slice(n, n_slices=1, offset=0):
    """ Splits a vector with ngram_size elements equally into n_slices
    returning a list of index ranges for that vector, each range corresponds
    to a slice.

    Args:
        n: number of elements in the vector
        offset: an offset for the slices
        n_slices: number of slices the vector is to be split into

    Returns:
        a list of slices for the vector
    """
    len_split = int(n / n_slices)

    ss = [0]
    for s in range(len_split, len_split * n_slices, len_split):
        ss.append(s)

    ss.append(n)
    ranges = [range(s[0] + offset, s[1] + offset) for s in _pairwise(ss)]

    return ranges


class Window:
    """ A Window used as utility to return windows around certain elements surrounded by
    other elements.

    A window contains:
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


def windows(seq, window_size=1):
    """ Transforms a sequence of strings to a sequence of windows

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


def flatten_it(iterable):
    """Flatten one level of nesting in an iterator"""
    return itertools.chain.from_iterable(iterable)


def take_it(n, iterable):
    """Return first n items of the iterable as a list"""
    return itertools.islice(iterable, n)


def chunk_it(dataset, n_rows=None, chunk_size=1):
    """
    Allows to iterate over dataset by loading chunks at a time using slices
    up until a given nrows

    :param dataset: the dataset we wish to iterate over
    :param n_rows: number of rows we want to take from the dataset (start at 0)
    :param chunk_size: the chunk size to be loaded into the memory
    :return: and iterator over the elements of dataset with buffered slicing
    """
    if n_rows is None:
        try:
            n_rows = len(dataset)
        except TypeError:
            raise TypeError("n_rows is None but dataset has no len()")

    if chunk_size > n_rows:
        chunk_size = n_rows

    n_chunks = n_rows // chunk_size
    chunk_slices = divide_slice(n_rows, n_chunks)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    # row_gen = itertools.chain.from_iterable((c[i] for i in range(len(c))) for c in chunk_gen)
    row_gen = itertools.chain.from_iterable(chunk_gen)
    return row_gen


def subset_chunk_it(dataset, data_range, chunk_size=1):
    """Allows to iterate over a given subset of a given dataset by loading chunks at a time

    dataset: the given dataset
    data_range: a range from which we will extract the ngram_size chunks to be loaded from the dataset
    chunk_size: length of each chunk to be loaded, this determines the number of chunks

    """
    n_rows = len(data_range)

    if chunk_size > n_rows:
        chunk_size = n_rows

    n_chunks = n_rows // chunk_size
    chunk_slices = divide_slice(n_rows, n_chunks, data_range.start)
    chunk_gen = (dataset[slice(s.start, s.stop, 1)] for s in chunk_slices)

    row_gen = itertools.chain.from_iterable((c[i] for i in range(len(c))) for c in chunk_gen)
    return row_gen


def n_grams(seq, n=1):
    """ Creates a list of n-grams from a list of strings

    :param seq: list of strings
    :param n: size for the n-gram
    :return:
    """
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def batch_it(data_it, size, padding=False, padding_elem=None):
    """

    Args:
        padding_elem: the element to be used to pad the batches
        padding: if True, pads the last batch to be of at least the given size
        data_it: an iterable over data to be batched
        size: size of the batch to be returned

    Returns:
        a generator over batches with a given size, these can be smaller


    """
    source_iter = iter(data_it)
    try:
        while True:
            batch_iter = take_it(size, source_iter)
            next_batch = list(itertools.chain([next(batch_iter)], batch_iter))
            if padding and len(next_batch) < size:
                padding_size = size - len(next_batch)
                next_batch.extend([padding_elem] * padding_size)
            yield next_batch
    except StopIteration:
        return


def shuffle_it(data_it, buffer_size):
    """ Shuffle iterator based on a buffer size

    Shuffling requires a finite list so we can use a buffer to build a list

    Args:
        data_it: an iterable over data
        buffer_size: the size of

    Returns:

    """
    buffer_it = batch_it(data_it, size=buffer_size)
    result = map(np.random.permutation, buffer_it)
    shuffled = itertools.chain.from_iterable((elem for elem in result))

    return shuffled


def chain_it(*data_it):
    """ Forward method for itertools chain

    To avoid unnecessary imports when using the "recipes" in views

    Args:
        *data_it: the iterables to be chained

    Returns:
        returns elements from the first iterable until exhausted, proceeds to the following iterables untill all are
        exhausted.

    """
    return itertools.chain(*data_it)


def repeat_fn(fn, data, n):
    """ Repeats the iter_fn on the given data, n times

    Warning: this intended to create iterators that cycle multiple times though
    data without having to copy the elements to cycle through them. If the
    fn returns a generator that iterates in a certain manner, this re-applies
    that same generator to the data. If however, data is an iterable that gets
    exhausted the first time it runs, this will return all the elements in the iterable
    just once.

    Args:
        data: the data to which the iter fn is to be applied
        fn : a function to be applied to the data
        n: number of times we iterate over iterable

    Returns:
        a generator on elems in data given by the iter_fn it

    """
    iter_it = (fn(data) for fn in itertools.repeat(fn, n))

    return itertools.chain.from_iterable(iter_it)
