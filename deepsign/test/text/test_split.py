import unittest
import deepsign.utils.views as split


class TestSplit(unittest.TestCase):

    def test_slice(self):
        v = range(0,16,1)
        num_splits = 5

        slices = split.divide_slice(len(v),num_splits)
        self.assertEqual(len(slices),num_splits)

        print(slices)


if __name__ == '__main__':
    unittest.main()