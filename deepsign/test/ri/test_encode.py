import unittest
import numpy.testing as np_test
import numpy as np
from deepsign.ri.encode import BoW,BoWDir
from deepsign.ri.sign_index import SignIndex
from deepsign.ri.core import RandomIndexGenerator
from deepsign.ri.permutations import PermutationGenerator
from deepsign.text.windows import sliding_windows


class TestEncode(unittest.TestCase):

    def setUp(self):
        dim = 10
        act = 2
        self.generator = RandomIndexGenerator(dim=dim, active=act)
        self.sign_index = SignIndex(self.generator)
        self.perm_generator = PermutationGenerator(dim=dim)

    def test_BoW_create(self):
        data = ["A", "B", "A", "C", "A", "B"]

        for s in data:
            self.sign_index.add(s)

        unique_str = set(data)
        self.assertEqual(self.sign_index.size(),len(unique_str))

        encoder = BoW(self.sign_index, window_size=1)
        vectors = encoder.encode(data)

    def test_BoW_ignore_order(self):
        data1 = ["A", "B"]
        data2 = ["B", "A"]

        for s1,s2 in data1,data2:
            self.sign_index.add(s1)
            self.sign_index.add(s2)

        encoder = BoW(self.sign_index, window_size=1)
        v1 = encoder.encode(data1)
        v2 = encoder.encode(data2)

        np_test.assert_array_equal(v1[0],v2[0])
        np_test.assert_array_equal(v1[1],v2[1])

        a_ri = self.sign_index.get_ri("A")
        b_ri = self.sign_index.get_ri("B")

        np_test.assert_array_equal(v1[0]- a_ri.to_vector(),
                                   b_ri.to_vector())

    def test_BoWDir_create(self):
        data0 = ["A"]
        data1 = ["A", "B","C"]
        data2 = ["A", "C","B"]

        for i in range(len(data1)):
            self.sign_index.add(data1[i])
            self.sign_index.add(data2[i])

        encoder = BoWDir(self.sign_index, window_size=2)

        v0 = encoder.encode(data0)
        np_test.assert_array_equal(v0[0],self.sign_index.get_ri("A").to_vector())

        v1 = encoder.encode(data1)
        self.assertEqual(len(v1),len(data1))
        self.assertEqual(len(v1),len(sliding_windows(data1)))

        v2 = encoder.encode(data2)

        # get encodings for first window of each string vector
        u = v1[0]
        v = v2[0]

        w1 = sliding_windows(data1,window_size=2)
        w2 = sliding_windows(data2,window_size=2)
        self.assertSetEqual(set(w1[0].right),set(w2[0].right))



        # first two windows should be the same
        np_test.assert_array_equal(u,v)

        #TODO refactor: encoders should work per window and not per sentence; different encoders can be applied to the whole sentence

if __name__ == '__main__':
    unittest.main()