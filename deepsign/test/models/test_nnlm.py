import unittest
import tensorflow as tf
from deepsign.models import nnlm

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()
    def tearDown(self):
        self.ss.close()

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
