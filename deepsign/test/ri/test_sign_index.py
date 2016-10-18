import unittest
from deepsign.rp.ri import RandomIndexGenerator, RandomIndex
from deepsign.rp.index import SignIndex


class TestSignIndex(unittest.TestCase):
    def test_size(self):
        gen = RandomIndexGenerator(100, 10)
        sign_index = SignIndex(generator=gen)

        # adding elements should increase size
        self.assertEqual(len(sign_index), 0)

        sign_index.add("0")
        self.assertEqual(len(sign_index), 1)
        self.assertEqual(sign_index.nextID, sign_index.get_id("0") + 1)

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

        gen = RandomIndexGenerator(dim, act)
        sign_index = SignIndex(generator=gen)

        sign_index.add("0")
        self.assertTrue(sign_index.contains("0"))
        self.assertFalse(sign_index.contains("1"))

        sign_index.remove("0")
        self.assertFalse(sign_index.contains("0"))

    def test_get(self):
        dim = 100
        act = 10

        gen = RandomIndexGenerator(dim, act)
        sign_index = SignIndex(gen)

        sign_index.add("0")
        ri0 = sign_index.get_ri("0")
        self.assertIsInstance(ri0, RandomIndex)

        self.assertEqual(ri0.dim, dim)


if __name__ == '__main__':
    unittest.main()
