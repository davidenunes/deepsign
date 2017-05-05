from unittest import TestCase
import tensorflow as tf

from tensorx.layers import Input, SparseInput, Dense, Act, Embeddings, Merge, FeatureInput
from tensorx.init import glorot_init
import numpy as np
import tensorx.utils.io as txio
import random
from tensorx.utils import transform

def generate(dim, num_active):
    active_indexes = random.sample(range(dim), num_active)

    num_positive = num_active // 2
    positive = active_indexes[0:num_positive]
    negative = active_indexes[num_positive:len(active_indexes)]

    return (positive, negative)


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

        (positive, negative) = generate(dim, n_active)
        ri_indexes = positive + negative
        ri_indexes = np.asmatrix(ri_indexes)

        ri_vector = np.zeros(dim)
        ri_vector[ri_indexes] = 1
        ri_vector = np.asmatrix(ri_vector)

        input_ids = SparseInput(n_units=dim, n_active=n_active, dtype=tf.int32)
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
        (positive, negative) = generate(dim, n_active)
        ri_indexes = positive, negative
        ri_indexes = np.asmatrix(ri_indexes)

        ri_vector = np.zeros(dim)
        ri_vector[ri_indexes] = 1
        ri_vector = np.asmatrix(ri_vector)

        # network
        pos_input = FeatureInput(n_units=dim, n_active=len(positive), dtype=tf.int32)
        neg_input = FeatureInput(n_units=dim, n_active=len(negative), dtype=tf.int32)

        pos_features = Embeddings(pos_input, h, bias=False)
        neg_features = Embeddings(neg_input, h, weights=pos_features.weights, bias=False)
        out1 = Merge([pos_features, neg_features], weights=[1, -1], bias=True)

        init = tf.global_variables_initializer()
        feed = {pos_input(): [positive], neg_input(): [negative]}

        # saver = tf.train.Saver()


        with tf.Session() as ss:
            ss.run(init)

            print("pos: ", positive)
            print("neg: ", negative)

            w = ss.run(pos_features.weights, feed_dict=feed)
            print("w:\n", w)

            out = ss.run(out1(), feed_dict=feed)
            print("selected: \n", out)

            txio.save_graph(ss, "/home/davex32/tmp")

    def test_sparse_input(self):
        dim = 10
        active = 4

        (positive, negative) = generate(dim, active)
        indices = positive + negative
        values = np.random.rand(active)

        indices = [indices]

        shape = np.array([1, dim], dtype=np.int64)
        indices = transform.indices_to_sparse(indices, shape)
        values = transform.values_to_sparse(values, indices.indices, shape)

        sp_input = SparseInput(n_units=dim,values=True)
        with tf.Session() as ss:
            result = ss.run(tf.scalar_mul(2,sp_input()[0]), feed_dict={sp_input.indices: indices,
                                                   sp_input.values: values})

            print(result)
