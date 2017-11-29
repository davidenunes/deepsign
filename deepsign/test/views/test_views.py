from unittest import TestCase
import numpy as np
from deepsign.utils.views import chunk_it, subset_chunk_it
from deepsign.utils.views import divide_slice, ngram_windows



class TestViews(TestCase):

    def test_chunk_it(self):
        nrows = 100

        data = np.arange(nrows)
        it = chunk_it(data, nrows, 3)

        for i in range(len(data)):
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_subset_chunk_it(self):
        nrows = 100

        data = np.arange(nrows)
        subset = range(50,100)

        it = subset_chunk_it(data, subset, 4)

        for i in subset:
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_divide_slice(self):
        data = range(100)
        subset = range(51,100)

        subsize = len(subset)
        divide_sub = divide_slice(subsize,3,subset.start)

        self.assertEqual(3, len(divide_sub))

    def test_ngrams(self):
        sentence = "hello there mr smith welcome back to the world"
        tokens = sentence.split()
        windows = ngram_windows(tokens,3)
        for window in windows:
            print(window)

        print("fewer than ngram_size sequences")
        sentence = "hello there"

        tokens = sentence.split()
        windows = ngram_windows(tokens, 3)
        self.assertEqual(len(windows),0)

        for window in windows:
            print(window)
