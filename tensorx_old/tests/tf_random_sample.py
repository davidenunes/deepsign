from unittest import TestCase
import random
import tensorflow as tf
import numpy as np

class TestRandomSample(TestCase):
    def test_python_impl(self):
        def sample_n(ary,n):
            print(ary)
            result = []
            for d in range(0, n, 1):
                e = random.randint(d, len(ary)-1)

                ary[d], ary[e] = ary[e], ary[d]
                result.append(ary[d])
            print(ary)
            return result

        n = np.arange(0,10)
        print(sample_n(n,3))


    def test_tf_impl(self):
        # 1 generate n random indexes
        # 2 gather range from generated indexes


        with tf.Session() as ss:

            n = 2

            r = tf.range(0,10)

            def gen(i):

                a = tf.add(i,1)
                return a, tf.random_uniform([1],minval=i,maxval=10,dtype=tf.int32)

            i = tf.constant(0)

            cond = lambda i,_: tf.less(i, n)



            r = tf.while_loop(cond,gen,[i,_])
            print(ss.run(r))
