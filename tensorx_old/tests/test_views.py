from unittest import TestCase
from tensorx_old.utils import transform
import numpy as np

class TestViews(TestCase):
    def test_indices_to_sparse(self):
        indices = [[0], [0, 5], [0, 1, 2]]
        shape = [3, 10]

        sp_indices = transform.indices_to_sparse(indices, shape)
        self.assertEqual(len(sum(indices, [])), len(sp_indices.indices))
        self.assertEqual(len(sum(indices, [])), len(sp_indices.values))

        indices = [[0, 10]]

        self.assertRaises(Exception, transform.indices_to_sparse, indices, shape)

        try:
            indices = [[0, 9]]
        except Exception as e:
            self.fail("Should not fail with valid indices: ", e)

    def test_values_to_sparse(self):
        indices = [[0], [0, 5], [0, 1, 2]]
        shape = [3, 10]
        values = np.random.rand(6,1)
        sp_indices = transform.indices_to_sparse(indices, shape)
        sp_values = transform.values_to_sparse(values, sp_indices.indices, shape)

