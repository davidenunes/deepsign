from unittest import TestCase
import numpy as np
from deepsign.data import transform


class TestTransform(TestCase):
    def test_batch_one_hot(self):
        shape = [3, 4]
        indices = [1, 3, 2]

        v = np.zeros(shape)
        b_index = np.arange(shape[0])
        b_index *= shape[1]
        v.put(indices + b_index, 1)

        one_hot = transform.batch_one_hot(indices, shape[1])

        np.testing.assert_array_equal(one_hot, v)

        indices = np.reshape(indices, [shape[0], 1])

        one_hot2 = transform.batch_one_hot(indices, shape[1])
        np.testing.assert_array_equal(one_hot2, v)
