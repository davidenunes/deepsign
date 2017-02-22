import unittest
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.index import TrieSignIndex


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

    def test_get(self):
        dim = 100
        act = 10

        gen = Generator(dim, act)
        sign_index = TrieSignIndex(gen)

        sign_index.add("0")
        self.assertTrue(sign_index.contains("0"))
        ri0 = sign_index.get_ri("0")
        self.assertIsInstance(ri0, RandomIndex)

        self.assertEqual(ri0.dim, dim)


if __name__ == '__main__':
    unittest.main()