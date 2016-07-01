import unittest
from deepsign.rp import ri


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

if __name__ == '__main__':
    unittest.main()
