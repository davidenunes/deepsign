from unittest import TestCase
import numpy as np
from deepsign.data.views import chunk_it, subset_chunk_it
from deepsign.data.views import divide_slice, n_grams, batch_it, shuffle_it, flatten_it, repeat_fn, chain_it
import itertools


class TestViews(TestCase):

    def test_chunk_it(self):
        n_rows = 100

        data = np.arange(n_rows)
        it = chunk_it(data, n_rows, 3)

        for i in range(len(data)):
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_subset_chunk_it(self):
        n_rows = 100

        data = np.arange(n_rows)
        subset = range(50, 100)

        it = subset_chunk_it(data, subset, 4)

        for i in subset:
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_divide_slice(self):
        subset = range(51, 100)

        sub_size = len(subset)
        divide_sub = divide_slice(sub_size, 3, subset.start)

        self.assertEqual(3, len(divide_sub))

    def test_ngrams(self):
        sentence = "hello there mr smith welcome back to the world"
        tokens = sentence.split()
        windows = n_grams(tokens, 3)
        for window in windows:
            print(window)

        print("fewer than ngram_size sequences")
        sentence = "hello there"

        tokens = sentence.split()
        windows = n_grams(tokens, 3)
        self.assertEqual(len(windows), 0)

        for window in windows:
            print(window)

    def test_batch_it(self):
        num_samples = 6
        v = np.random.uniform(-1, 1, [num_samples, 2])
        padding = np.zeros([2])

        c_it = chunk_it(v, 6, chunk_size=3)
        print(v)

        batch_size = 4
        b_it = batch_it(c_it, batch_size, padding=True, padding_elem=padding)

        for b in b_it:
            self.assertEqual(len(b), batch_size)
            print(np.array(b))

        b_it = batch_it(v, batch_size)
        last_batch = None
        try:
            for b in b_it:
                last_batch = b
                self.assertEqual(len(b), batch_size)

        except AssertionError:
            self.assertEqual(len(last_batch), num_samples % batch_size)

    def test_shuffle_it(self):
        v = list(range(10))
        padding = -1

        b_it = batch_it(v, size=4, padding=True, padding_elem=padding)

        s_it = shuffle_it(b_it, 3)
        for elem in s_it:
            print(elem)

    def test_reat_chunk_it(self):
        n_samples = 4
        repeat = 2
        v = np.random.uniform(0, 1, [n_samples, 1])
        data_it = chunk_it(v, chunk_size=2)

        def chunk_fn(x): return chunk_it(x, chunk_size=2)

        # for chunk in data_it:
        #    print(chunk)
        # print(data_it)
        data_it = repeat_fn(chunk_fn, v, repeat)

        self.assertEqual(len(list(data_it)), n_samples * repeat)

    def test_chain_shuffle(self):
        n_samples = 4
        repeat = 2
        v = np.arange(0, n_samples, 1)
        data_it = chunk_it(v, chunk_size=2)

        def chunk_fn(x): return chunk_it(x, chunk_size=2)

        # first chain is normal, second is shuffled from the two repetitions
        data_it = repeat_fn(chunk_fn, v, repeat)

        data_it = chain_it(data_it, shuffle_it(repeat_fn(chunk_fn, v, repeat), buffer_size=8))

        data = list(data_it)

        unique_data = np.unique(data)
        counts = np.unique(np.bincount(data))

        self.assertEqual(len(unique_data), 4)
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0], 4)

    def test_repeat_fn_exhaust(self):
        n_samples = 4
        repeat = 2
        v = np.random.uniform(0, 1, [n_samples, 1])
        data_it = chunk_it(v, chunk_size=2)

        def it_fn(x): return iter(x)

        # data it will get exhausted so it will not repeat
        data_it = repeat_fn(it_fn, data_it, repeat)

        # only return 4 items
        self.assertEqual(len(list(data_it)), n_samples)
