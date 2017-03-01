import unittest
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.index import TrieSignIndex
import os


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




        filename = "index"
        dir = os.getwd()



        os.path.isfile





if __name__ == '__main__':
    unittest.main()