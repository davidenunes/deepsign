import unittest
from deepsign.rp.ri import Generator
import numpy as np
import numpy.testing as npt

import deepsign.data.views as view


class TestSparseNPView(unittest.TestCase):
    def setUp(self):
        dim = 1000
        act = 10
        self.generator = Generator(dim=dim, active=act)

    def test_ri_to_sparse(self):
        ri_vector = self.generator.generate().to_vector()
        dim = ri_vector.shape[0]

        compact1 = view.np_to_sparse(ri_vector)

        self.assertEqual(compact1.dim, dim)
        npt.assert_array_equal(compact1.active, np.nonzero(ri_vector))

        rebuilt = compact1.to_vector()
        npt.assert_array_equal(rebuilt, ri_vector)
