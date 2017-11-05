from unittest import TestCase
import numpy as np
import tensorflow as tf

class TestCreateNoise(TestCase):

    def test_uniform_sampler(self):

        sample = tf.nn.uniform_candidate_sampler(
            true_classes=tf.reshape(tf.cast(tf.range(0,1,1),tf.int64),shape=[1,1]),
            num_true=1,
            num_sampled=3,
            unique=True,
            range_max=10)
        #sample = tf.nn.uniform_candidate_sampler(true_classes=tf.zeros([1],tf.int64),num_true=1, num_sampled=2,unique=True,range_max=100)
        sample = tf.random_uniform([10],minval=0,maxval=10,dtype=tf.int64)
        drop = tf.nn.dropout(tf.ones([10]),keep_prob=0.5)

        with tf.Session() as ss:

            perm = tf.map_fn(
                lambda i: tf.random_uniform([1],
                                            minval=i,
                                            maxval=10,
                                            dtype=tf.int64),
                tf.cast(tf.range(0,4),tf.int64))
            perm = tf.reshape(perm,[-1])


            print(ss.run(perm))
