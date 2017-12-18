from unittest import TestCase
import numpy as np
from deepsign.data.views import chunk_it, subset_chunk_it
from deepsign.data.views import divide_slice, n_grams


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
