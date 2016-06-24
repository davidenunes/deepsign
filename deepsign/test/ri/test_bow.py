import unittest
from deepsign.ri.encode import BoWEncoder
from deepsign.ri.sign_index import SignIndex
from deepsign.ri.core import RandomIndexGenerator


class TestBoW(unittest.TestCase):

    def test_create_encoder(self):
        data = ["A", "B", "A", "C", "A", "B"]

        generator = RandomIndexGenerator(4, 2)
        sign_index = SignIndex(generator)

        for s in data:
            sign_index.add(s)

        unique_str = set(data)
        self.assertEqual(sign_index.size(),len(unique_str))

        encoder = BoWEncoder(sign_index,window_size=1)

        vectors = encoder.encode(data)

        for v in vectors: print(v)

        # TODO :do an exhaustive test; but everything seems to be working, there are no "edge" cases I think


if __name__ == '__main__':
    unittest.main()