from unittest import TestCase
import numpy as np
from deepsign.utils.views import chunk_it


class TestViews(TestCase):

    def test_chunk_it(self):
        nrows = 100

        data = np.arange(nrows)
        it = chunk_it(data, nrows, 3)

        for i in range(len(data)):
            data_j = next(it)
            self.assertEqual(data[i], data_j)