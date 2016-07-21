import unittest
from deepsign.rp import ri
import numpy as np
import numpy.testing as npt


class TestRI(unittest.TestCase):

    def test_generator(self):
        dim = 18
        active = 2

        gen = ri.RandomIndexGenerator(dim=dim, active=active)

        ri1 = gen.generate()

        self.assertEqual(len(ri1.negative), len(ri1.negative))
        self.assertEqual(len(ri1.positive),gen.num_active//2)

        v1 = ri1.to_vector()

        self.assertEqual(len(v1),dim)
        self.assertEqual(v1.max(),1)
        self.assertEqual(v1.min(),-1)
        self.assertEqual(v1.sum(),0)

        vectors = [gen.generate().to_vector() for x in range(0, 18)]

        for v in vectors: print(v)

    def test_rescale(self):
        dim = 10
        active = 4

        gen = ri.RandomIndexGenerator(dim=dim, active=active)

        ri1 = gen.generate().to_vector()
        ri2 = gen.generate().to_vector()

        s = ri1 + ri1 + ri2

        print(s)

        print(s/np.max(s,axis=0))

    def test_to_sparse(self):
        dim = 100
        active = 4

        gen = ri.RandomIndexGenerator(dim=dim, active=active)

        index1 = gen.generate()
        ri_v1 = index1.to_vector()
        sparse_array = index1.to_sparse()
        self.assertEqual(len(sparse_array),active)

        index2 = ri.from_sparse(dim,active,sparse_array)
        ri_v2 = index2.to_vector()

        npt.assert_array_equal(ri_v1,ri_v2)



if __name__ == '__main__':
    unittest.main()
