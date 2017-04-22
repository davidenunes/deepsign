from unittest import TestCase
import numpy as np
import tensorflow as tf

class TestCreateNoise(TestCase):

    def test_uniform_sampler(self):


        sample = tf.random_uniform([10],minval=0,maxval=10,dtype=tf.int64)
        drop = tf.nn.dropout(tf.ones([10]),keep_prob=0.5)

        with tf.Session() as ss:
            r = ss.run(drop)
            print(r)
