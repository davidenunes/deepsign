import unittest
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.index import TrieSignIndex
import os
import h5py
import numpy as np

class TestTrieSignIndex(unittest.TestCase):
    def test_size(self):
        gen = Generator(100, 10)
        sign_index = TrieSignIndex(generator=gen)

        # adding elements should increase size
        self.assertEqual(len(sign_index), 0)

        sign_index.add("0")
        self.assertEqual(len(sign_index), 1)

        # duplicated elements are not added
        sign_index.add("0")
        self.assertEqual(len(sign_index), 1)

        sign_index.add("1")
        self.assertEqual(len(sign_index), 2)

        # removing elements should reduce size
        size_before = len(sign_index)

        sign_index.remove("0")
        size_after = len(sign_index)
        self.assertEqual(size_after, size_before - 1)

    def test_contains(self):
        dim = 100
        act = 10

        gen = Generator(dim, act)
        sign_index = TrieSignIndex(generator=gen)

        sign_index.add("0")
        self.assertTrue(sign_index.contains("0"))
        self.assertFalse(sign_index.contains("1"))

        sign_index.remove("0")

        self.assertFalse(sign_index.contains("0"))

    def test_get_ri(self):
        dim = 100
        act = 10

        gen = Generator(dim, act)
        sign_index = TrieSignIndex(gen)

        sign_index.add("0")
        self.assertTrue(sign_index.contains("0"))
        ri0 = sign_index.get_ri("0")
        self.assertIsInstance(ri0, RandomIndex)

        self.assertEqual(ri0.dim, dim)

    def test_get_sign(self):
        dim = 100
        act = 10
        gen = Generator(dim, act)

        signs = [str(i) for i in range(10)]
        sign_index = TrieSignIndex(gen, vocabulary=signs)

        for s in signs:
            self.assertTrue(sign_index.contains(s))
            id = sign_index.get_id(s)
            self.assertTrue(sign_index.contains_id(id))
            s2 = sign_index.get_sign(id)
            self.assertEqual(s,s2)\



        #get sign for an id that doesn't exist
        id = 86
        s = sign_index.get_sign(id)
        self.assertEqual(s,None)
        self.assertFalse(sign_index.contains_id(id))

        self.assertEqual(len(sign_index.sign_trie),len(signs))

        self.assertTrue(sign_index.contains_id(len(signs)-1))
        self.assertFalse(sign_index.contains_id(len(signs)))

        #self.assertTrue(sign_index)

    def test_save(self):
        dim = 100
        act = 10
        gen = Generator(dim, act)

        signs = [str(i) for i in range(10)]
        sign_index = TrieSignIndex(gen, vocabulary=signs)

        filename = "index.hdf5"
        directory = os.path.dirname(os.path.abspath(__file__))
        output_file = directory+"/"+filename

        self.assertFalse(os.path.exists(output_file))
        try:
            sign_index.save(output_file)
            self.assertTrue(os.path.exists(output_file))

            h5file = h5py.File(output_file,'r')

            h5signs = h5file["signs"]
            h5ri = h5file["ri"]

            self.assertEqual(len(h5signs),len(signs))

            print(h5ri[0])
            print(h5ri.attrs["k"])
            print(h5ri.attrs["s"])
            print(h5ri.attrs["state"].tostring())

            h5file.close()
        except:
            raise
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
        self.assertFalse(os.path.exists(output_file))

    def test_load(self):
        """ The ids should be the same when the index is loaded back up

        """
        dim = 100
        act = 10
        gen = Generator(dim, act)

        signs1 = [str(i) for i in range(1000)]
        index1 = TrieSignIndex(gen, vocabulary=signs1)

        filename = "index.hdf5"
        directory = os.path.dirname(os.path.abspath(__file__))
        index_file = directory + "/" + filename

        self.assertFalse(os.path.exists(index_file))
        try:
            index1.save(index_file)
            self.assertTrue(os.path.exists(index_file))

            index2 = TrieSignIndex.load(index_file)
            self.assertEqual(len(index2),len(index1))

            for sign in signs1:
                self.assertTrue(index1.contains(sign))
                self.assertTrue(index2.contains(sign))
                id1 = index1.get_id(sign)
                id2 = index2.get_id(sign)
                self.assertEqual(id1,id2)

                ri1 = index1.get_ri(sign).to_vector()
                ri2 = index2.get_ri(sign).to_vector()

                np.testing.assert_array_equal(ri1,ri2)
        except:
            raise
        finally:
            if os.path.exists(index_file):
                os.remove(index_file)
        self.assertFalse(os.path.exists(index_file))


if __name__ == '__main__':
    unittest.main()