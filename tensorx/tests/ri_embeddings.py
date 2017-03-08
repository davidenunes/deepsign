from unittest import TestCase
from deepsign.rp.ri import Generator
import tensorflow as tf
from tensorx.models.nrp import NRP
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

        h_init = partial(tf.random_uniform,minval=-1, maxval=1)
        model = NRP(k_dim=k, h_dim=300, h_init=h_init)
        self.gen = gen
        self.model = model
        self.k = k

    def test_matmul_equals_lookup(self):
        input_id = Input(1,dtype=tf.int32)


        ss = tf.Session()

        # init model variables
        ss.run(tf.global_variables_initializer())

        w = ss.run(self.model.h.weights)

        #to get a single vector
        #embedding_aggregated = tf.reduce_sum(embedding_layer, [1])

        test_index = [[50]]

        lookup = tf.nn.embedding_lookup(params=self.model.h.weights,ids=input_id(),name="W_f")
        emb50 = ss.run(lookup,feed_dict={input_id(): test_index})
        emb50 = emb50[0]

        emb50_2 = ss.run(tf.gather(self.model.h.weights,input_id()),feed_dict={input_id():[[50]]})
        emb50_2 = emb50_2[0]

        np.testing.assert_array_equal(emb50, emb50_2)

        v = np.zeros(self.k)
        v[50] = 1
        emb50_3 =ss.run(self.model.h(),feed_dict={self.model.input(): [v]})

        np.testing.assert_array_equal(emb50, emb50_3)
        ss.close()

    def test_ri_equals_reducesum_lookup(self):

        ri = self.gen.generate()


        input_id = Input(len(ri.positive), dtype=tf.int32)




        ss = tf.Session()
        # init model variables
        ss.run(tf.global_variables_initializer())
        lookup = tf.nn.embedding_lookup(params=self.model.h.weights, ids=input_id(), name="W_f")
        embedding_v = tf.reduce_sum(lookup,axis=1)

        emb1_pos = ss.run(embedding_v,feed_dict={input_id(): [ri.positive]})
        emb1_neg = ss.run(embedding_v, feed_dict={input_id(): [ri.negative]})
        emb1 = emb1_pos - emb1_neg

        emb2 = ss.run(self.model.h(),feed_dict={self.model.input(): [ri.to_vector()]})

        np.testing.assert_array_equal(emb1, emb2)





