from unittest import TestCase
import tensorflow as tf
from typing import re
from deepsign.rp.ri import Generator

from tensorx.layers import Input, FeatureInput, Dense, Act, Embeddings, Merge
from tensorx.init import glorot_init
import numpy as np
import tensorx.utils.io as txio


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
        dense11 = Dense(input, 2, act=Act.sigmoid, name="layer11", init=glorot_init)
        dense12 = Dense(input, 2, weights=dense11.weights, name="layer12", init=glorot_init)

        dense2 = Dense(input, 2, name="layer2", init=tf.ones)
        dense3 = Dense(dense2, 4, name="layer3", init=tf.ones)

        init = tf.global_variables_initializer()
        with tf.Session() as ss:
            ss.run(init)

            rand_input = np.random.rand(1, input_dim).astype(np.float32)
            res_input = ss.run(input(), feed_dict={input(): rand_input})

            np.testing.assert_equal(res_input, rand_input)

            res11 = ss.run(dense11(), feed_dict={input(): rand_input})
            res12_sig = ss.run(tf.sigmoid(dense12()), feed_dict={input(): rand_input})
            np.testing.assert_array_equal(res11, res12_sig)

            res2 = ss.run(dense2(), feed_dict={input(): rand_input})
            res3 = ss.run(dense3(), feed_dict={input(): rand_input})
            self.assertEqual(res2[0][0] * 2, res3[0][0])

    def test_embeddings(self):
        n_active = 2
        dim = 8

        # create a random input to simulate a RI
        gen = Generator(active=n_active, dim=dim)
        ri = gen.generate()
        ri_indexes = ri.positive + ri.negative
        ri_indexes = np.asmatrix(ri_indexes)

        ri_vector = np.zeros(dim)
        ri_vector[ri_indexes] = 1
        ri_vector = np.asmatrix(ri_vector)

        input_ids = FeatureInput(n_units=dim,n_active=n_active, dtype=tf.int32)
        input_full = Input(dim, dtype=tf.float32)

        dense = Dense(input_full, 2, act=Act.sigmoid, bias=True, name="dense")
        embeddings = Embeddings(input_ids, 2,
                                weights=dense.weights,
                                act=Act.sigmoid,
                                bias=True,
                                name="embed")

        init = tf.global_variables_initializer()

        with tf.Session() as ss:
            # init model variables
            ss.run(init)

            w1 = ss.run(dense.weights)
            w2 = ss.run(embeddings.weights)

            np.testing.assert_array_equal(w1, w2)

            result1 = ss.run(dense(), feed_dict={input_full(): ri_vector})
            result2 = ss.run(embeddings(), feed_dict={input_ids(): ri_indexes})

            np.testing.assert_array_equal(result1, result2)

    def test_merge(self):
        n_active = 2
        dim = 4
        h = 2

        # create a random input to simulate a RI
        gen = Generator(active=n_active, dim=dim)
        ri = gen.generate()
        ri_indexes = ri.positive + ri.negative
        ri_indexes = np.asmatrix(ri_indexes)

        ri_vector = np.zeros(dim)
        ri_vector[ri_indexes] = 1
        ri_vector = np.asmatrix(ri_vector)

        # network
        pos_input = FeatureInput(n_units=dim, n_active=len(ri.positive), dtype=tf.int32)
        neg_input = FeatureInput(n_units=dim, n_active=len(ri.negative), dtype=tf.int32)


        pos_features = Embeddings(pos_input,h,bias=False)
        neg_features = Embeddings(neg_input,h,weights=pos_features.weights,bias=False)
        out1 = Merge([pos_features,neg_features],weights=[1,-1],bias=True)

        init = tf.global_variables_initializer()
        feed = {pos_input(): [ri.positive], neg_input(): [ri.negative]}

        #saver = tf.train.Saver()


        with tf.Session() as ss:
            ss.run(init)

            print("pos: ",ri.positive)
            print("neg: ",ri.negative)

            w = ss.run(pos_features.weights,feed_dict=feed)
            print("w:\n", w)


            out = ss.run(out1(),feed_dict=feed)
            print("selected: \n", out)

            txio.save_graph(ss,"/home/davex32/tmp")










