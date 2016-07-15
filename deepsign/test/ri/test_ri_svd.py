import unittest
from deepsign.rp import ri
import numpy as np


class TestRISVD(unittest.TestCase):

    def test_svd(self):
        dim = 500
        active = 5
        gen = ri.RandomIndexGenerator(dim=dim, active=active)

        num_samples = 100

        c_matrix = np.matrix([gen.generate().to_vector() for i in range(num_samples)])

        c_matrix = np.matrix([ri / np.max(ri, axis=0) for ri in c_matrix])

        print("Original: ",c_matrix.shape)

        # perform svd
        u, s, vt = np.linalg.svd(c_matrix,full_matrices=False)
        print("Decomposition: ",(u.shape,np.diag(s).shape,vt.shape))

        # reconstruct
        r_matrix = np.dot(u, np.dot(np.diag(s),vt))
        print("Re-Construction: ",r_matrix.shape)
        self.assertTrue(np.allclose(c_matrix,r_matrix))

        # low-rank approximation
        k = 2
        ru = u[:,:k]
        rs = np.diag(s[:k])
        rvt = vt[:k]

        print("Low-Rank Decomposition: ", (ru.shape, rs.shape, rvt.shape))
        lr_matrix = np.dot(ru,np.dot(rs,rvt))
        print("Low-Rank Approximation Shape: ",lr_matrix.shape)
        self.assertEqual(lr_matrix.shape, c_matrix.shape)

        # dimensional reduction (just take u and s, since v is used to convert back to the original matrix)
        ld_matrix = np.dot(ru,rs)
        print("Lower-Dimensional Matrix: ",ld_matrix.shape)


if __name__ == '__main__':
    unittest.main()