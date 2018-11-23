import unittest

import numpy.testing as np_test

import deepsign.rp.encode as enc
from deepsign.rp.index import SignIndex
from deepsign.rp.permutations import PermutationGenerator
from deepsign.rp.ri import Generator
from deepsign.data.iterators import windows
import numpy as np


class TestEncode(unittest.TestCase):
    def setUp(self):
        dim = 10
        act = 2
        self.generator = Generator(dim=dim, num_active=act)
        self.sign_index = SignIndex(self.generator)
        self.perm_generator = PermutationGenerator(dim=dim)

    def test_bow_create(self):
        data = ["A", "B", "A", "C", "A", "B"]

        for s in data:
            self.sign_index.add(s)

        unique_str = set(data)
        self.assertEqual(len(self.sign_index), len(unique_str))

        windows = windows(data, window_size=1)
        vectors = [enc.to_bow(w, self.sign_index) for w in windows]
        self.assertEqual(len(vectors), len(windows))

    def test_bow_normalise(self):
        data = ["A", "A"]

        for s in data:
            self.sign_index.add(s)

        unique_str = set(data)
        self.assertEqual(len(self.sign_index), len(unique_str))

        windows = windows(data, window_size=1)
        norm_bow = enc.to_bow(windows[0], self.sign_index,normalise=True,include_target=True)
        self.assertEqual(np.max(norm_bow),1)


        unorm_bow = enc.to_bow(windows[0], self.sign_index, normalise=False,include_target=True)
        self.assertEqual(np.max(unorm_bow),2)


    def test_bow_ignore_order(self):
        data1 = ["A", "B"]
        data2 = ["B", "A"]

        for s1, s2 in data1, data2:
            self.sign_index.add(s1)
            self.sign_index.add(s2)

        windows1 = windows(data1, window_size=1)
        windows2 = windows(data2, window_size=1)

        v1 = enc.to_bow(windows1[0], self.sign_index)
        v2 = enc.to_bow(windows2[0], self.sign_index)

        np_test.assert_array_equal(v1, v2)
        np_test.assert_array_equal(v1, v2)

        a_ri = self.sign_index.get_ri("A")
        b_ri = self.sign_index.get_ri("B")

        np_test.assert_array_equal(v1 - a_ri.to_vector(),
                                   b_ri.to_vector())

    def test_bow_dir_create(self):
        data1 = ["A", "B", "C"]
        data2 = ["A", "C", "B"]

        for i in range(len(data1)):
            self.sign_index.add(data1[i])
            self.sign_index.add(data2[i])

        w1 = windows(data1, window_size=2)
        w2 = windows(data2, window_size=2)

        perm = self.perm_generator.matrix()
        v1 = enc.to_bow_dir(w1[0], sign_index=self.sign_index, perm_matrix=perm)
        v2 = enc.to_bow_dir(w2[0], sign_index=self.sign_index, perm_matrix=perm)

        self.assertSetEqual(set(w1[0].right), set(w2[0].right))
        np_test.assert_array_equal(v1, v2)


if __name__ == '__main__':
    unittest.main()
