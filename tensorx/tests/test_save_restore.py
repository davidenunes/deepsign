from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensorx.models.nrp2 import NRP
import os


class TestSaveRestore(TestCase):
    def test_save(self):

        home = os.getenv("HOME")
        result_dir = home + "/data/results/"

        model = NRP(k_dim=1000, h_dim=500)
        directory = os.getcwd()
        filename = result_dir + "model"
        var_init = tf.global_variables_initializer()

        tf_session = tf.Session()
        tf_session.run(var_init)
        w1 = tf_session.run(model.h.weights)

        self.assertFalse(os.path.exists(filename))
        model.save(tf_session,filename)
        self.assertTrue(os.path.exists(filename+".meta"))

        tf_session.run(var_init)
        w2 = tf_session.run(model.h.weights)

        try:
            np.testing.assert_array_equal(w1,w2)
        except AssertionError:
            pass
        else:
            raise

        model.load(tf_session,filename)

        w2 = tf_session.run(model.h.weights)
        print(w2)
        np.testing.assert_array_equal(w1, w2)



        tf_session.close()

        #os.remove(directory)


