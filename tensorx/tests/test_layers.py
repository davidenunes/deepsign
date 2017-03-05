from unittest import TestCase
import tensorflow as tf
from typing import re

from tensorx.layers import Input, Dense, Act
from tensorx.init import glorot
import numpy as np


class TestLayers(TestCase):
    def test_input(self):
        input = Input(10)
        self.assertIsInstance(input, Input)
        self.assertIsInstance(input(), tf.Tensor)

        with tf.Session() as sess:
            ones = np.ones(shape=(2, 10))
            ones = np.asmatrix(ones)

            result = sess.run(input(), feed_dict={input(): ones})
            print(result)

    def test_dense(self):
        input_dim = 4
        input = Input(input_dim)
        dense11 = Dense(input, 2, act=Act.sigmoid, name="layer11", init=glorot)
        dense12 = Dense(input, 2, weights=dense11.weights, name="layer12")

        dense2 = Dense(input, 2, name="layer2", init=tf.ones)
        dense3 = Dense(dense2, 4, name="layer3", init=tf.ones)

        init = tf.global_variables_initializer()
        with tf.Session() as ss:
            ss.run(init)

            rand_input = np.random.rand(1, input_dim).astype(np.float32)
            res_input = ss.run(input(), feed_dict={input(): rand_input})

            np.testing.assert_equal(res_input,rand_input)

            res11 = ss.run(dense11(), feed_dict={input(): rand_input})
            res12_sig = ss.run(tf.sigmoid(dense12()), feed_dict={input(): rand_input})
            np.testing.assert_array_equal(res11, res12_sig)

            res2 = ss.run(dense2(), feed_dict={input(): rand_input})
            res3 = ss.run(dense3(), feed_dict={input(): rand_input})
            self.assertEqual(res2[0][0] * 2, res3[0][0])
