import unittest
from deepsign.rp.ri import Generator
from deepsign.rp.index import SignIndex
from deepsign.data.transform import ris_to_sp_tensor_value


class MyTestCase(unittest.TestCase):
    def setUp(self):
        dim = 10
        act = 4
        self.generator = Generator(dim=dim, num_active=act)
        self.sign_index = SignIndex(self.generator)

    def test_encode_sp_create(self):
        sentence = ["A", "B"]

        for word in sentence:
            self.sign_index.add(word)

        ris = []
        for word in sentence:
            ri = self.sign_index.get_ri(word)
            ris.append(ri)

        result = ris_to_sp_tensor_value(ris, self.sign_index.feature_dim())
        print(result)

    def test_encode_sp_positive(self):
        """
        Testing encoding for positive-only sparse random vectors

        """
        sentence = ["A", "B"]

        for word in sentence:
            self.sign_index.add(word)

        ris = []
        for word in sentence:
            ri = self.sign_index.get_ri(word)
            ris.append(ri)

        result = ris_to_sp_tensor_value(ris, self.sign_index.feature_dim(), all_positive=True)
        print(result)


if __name__ == '__main__':
    unittest.main()
