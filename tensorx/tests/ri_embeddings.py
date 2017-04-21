from unittest import TestCase
from deepsign.rp.ri import Generator
import tensorflow as tf
from tensorx.models.nrp2 import NRP
from functools import partial
from deepsign.utils.views import sliding_windows
from tensorx.layers import Input
import numpy as np


class TestRIEmbeddings(TestCase):
    def setUp(self):
        k = 1000
        s = 4

        # random index dimension
        k = 1000
        s = 10
        h_dim = 300
        gen = Generator(active=s, dim=k)

        h_init = partial(tf.random_uniform, minval=-1, maxval=1)
        model = NRP(k_dim=k, h_dim=300, h_init=h_init)
        self.gen = gen
        self.model = model
        self.k = k

    def test_matmul_equals_lookup(self):
        input_id = Input(1, dtype=tf.int32)

        ss = tf.Session()

        # init model variables
        ss.run(tf.global_variables_initializer())

        w = ss.run(self.model.h.weights)

        # to get a single vector
        # embedding_aggregated = tf.reduce_sum(embedding_layer, [1])

        test_index = [[50]]

        lookup = tf.nn.embedding_lookup(params=self.model.h.weights, ids=input_id(), name="W_f")
        emb50 = ss.run(lookup, feed_dict={input_id(): test_index})
        emb50 = emb50[0]

        emb50_2 = ss.run(tf.gather(self.model.h.weights, input_id()), feed_dict={input_id(): [[50]]})
        emb50_2 = emb50_2[0]

        np.testing.assert_array_equal(emb50, emb50_2)

        v = np.zeros(self.k)
        v[50] = 1
        emb50_3 = ss.run(self.model.h(), feed_dict={self.model.input(): [v]})

        np.testing.assert_array_equal(emb50, emb50_3)
        ss.close()

    def test_ri_equals_reducesum_lookup(self):
        ri = self.gen.generate()
        input_id = Input(len(ri.positive), dtype=tf.int32)
        ss = tf.Session()
        # init model variables
        ss.run(tf.global_variables_initializer())
        lookup = tf.nn.embedding_lookup(params=self.model.h.weights, ids=input_id(), name="W_f")
        embedding_v = tf.reduce_sum(lookup, axis=1)

        emb1_pos = ss.run(embedding_v, feed_dict={input_id(): [ri.positive]})
        emb1_neg = ss.run(embedding_v, feed_dict={input_id(): [ri.negative]})
        emb1 = emb1_pos - emb1_neg

        emb2 = ss.run(self.model.h(), feed_dict={self.model.input(): [ri.to_vector()]})

        np.testing.assert_array_equal(emb1, emb2)

    def test_sparse_gradient_lookup(self):
        """ Test if gradient updates with embedding lookup 
        have the same result of dense gradients using RI directly
        
        """

        n_rows = 8
        n_cols = 2

        n_active = 2

        full_input = tf.placeholder(dtype=tf.float32, shape=[1, n_rows])
        var1 = tf.Variable(tf.random_uniform(shape=[n_rows, n_cols], minval=-1, maxval=1), name="var1")

        output1 = tf.matmul(full_input, var1)

        id_input = tf.placeholder(dtype=tf.int32, shape=[None, n_active])
        var2 = tf.Variable(var1.initialized_value(), name="var2")
        lookup = tf.nn.embedding_lookup(params=var2, ids=id_input)

        # axis = 0 guarantees that we are summing the columns of each vector
        output2 = tf.reduce_sum(lookup, axis=1)

        # create a random objective with the same shape as the embeddings
        # the gradient updates should be the same with and without the embedding lookup

        target_output = np.random.uniform(low=-1, high=1, size=n_cols)
        target_output = np.asmatrix(target_output)

        # create a random input to simulate a RI
        gen = Generator(active=n_active, dim=n_rows)
        ri = gen.generate()
        ri_indexes = ri.positive + ri.negative
        ri_indexes = np.asmatrix(ri_indexes)

        ri_vector = np.zeros(n_rows)
        ri_vector[ri_indexes] = 1
        ri_vector = np.asmatrix(ri_vector)

        target_label = tf.placeholder(shape=[1, n_cols], dtype=tf.float32, name="labels")
        loss1 = tf.losses.mean_squared_error(labels=target_label, predictions=output1)
        loss2 = tf.losses.mean_squared_error(labels=target_label, predictions=output2)

        var1_grad = tf.gradients(loss1, var1)
        var2_grad = tf.gradients(loss2, var2)

        init = tf.global_variables_initializer()
        with tf.Session() as ss:
            ss.run(init)

            out1 = ss.run(output1, feed_dict={full_input: ri_vector})
            out2 = ss.run(output2, feed_dict={id_input: ri_indexes})
            np.testing.assert_array_equal(out1, out2)

            l1 = ss.run(tf.reduce_mean(tf.squared_difference(output1, target_output)),
                        feed_dict={full_input: ri_vector, target_label: target_output})
            l2 = ss.run(tf.reduce_mean(tf.squared_difference(output2, target_output)),
                        feed_dict={id_input: ri_indexes, target_label: target_output})

            self.assertEqual(l1,l2)

            # the gradient updates should be the same but the first produces dense gradient updates
            grads1 = ss.run(var1_grad, feed_dict={full_input: ri_vector, target_label: target_output})
            grads2 = ss.run(var2_grad, feed_dict={id_input: ri_indexes, target_label: target_output})

            all_grads1 = grads1[0][np.all(grads1[0],axis=1)]
            all_grads2 = grads2[0].values

            # gradient updates should be the same
            np.testing.assert_array_equal(all_grads1,all_grads2)

