import unittest
import deepsign.data.iterators as split


class TestSplit(unittest.TestCase):
    def test_slice(self):
        v = range(0, 16, 1)
        num_splits = 5

        slices = split.divide_slices(len(v), num_splits)
        self.assertEqual(len(slices), num_splits)

        print(slices)
        for r in slices:
            for elem in r:
                print(elem)
            print("-------")


if __name__ == '__main__':
    unittest.main()
