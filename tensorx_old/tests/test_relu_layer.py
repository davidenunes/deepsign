from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensorx_old.layers import Dense, Embeddings
from deepsign.rp.ri import Generator, RandomIndex


class TestReLU(TestCase):
    """
    Simple use case exploring how to take advantage of ReLU sparsity 
    The idea is to use relus to propagate sparse gradients
    """

    def test_relu_to_sparse(self):
        """ Test if gradient updates with embedding lookup 
            have the same result of dense gradients using RI directly

        """

        dim = 10
        h_dim = 6

        n_active = 2

        id_input = tf.placeholder(dtype=tf.int32, shape=[None, n_active])

        weights1 = tf.Variable(tf.random_uniform(shape=[dim, h_dim], minval=-1, maxval=1), name="weights1")
        lookup = tf.nn.embedding_lookup(params=weights1, ids=id_input)
        hx = tf.reduce_sum(lookup, axis=1)

        # convert relu to sparse
        h = tf.nn.relu(hx)

        def dense_to_sparse_ids(t):
            idx = tf.where(tf.not_equal(t, 0))
            rs_t = tf.reshape(t, [-1])
            values = tf.mod(tf.squeeze(tf.where(tf.not_equal(rs_t, 0))), h_dim)
            dense_shape = tf.cast(tf.shape(t),tf.int64)
            return tf.SparseTensor(idx, values, dense_shape=dense_shape)

        def dense_to_sparse_values(t):
            idx = tf.where(tf.not_equal(t, 0))
            values = tf.gather_nd(params=t, indices=idx)
            dense_shape = tf.cast(tf.shape(t), tf.int64)
            return tf.SparseTensor(idx, values, dense_shape=dense_shape)

        def positive_idx(t):
            return tf.where(tf.not_equal(t,0))



        weights2 = tf.Variable(tf.random_uniform(shape=[h_dim,dim], minval=-1, maxval=1), name="weights2")
        #output = tf.matmul(h,weights2)
        output = tf.nn.embedding_lookup_sparse(params=weights2,
                                                       sp_ids=dense_to_sparse_ids(h),
                                                       sp_weights=None,
                                                       combiner="sum"
                                                       #sp_weights=dense_to_sparse_values(h)
                                                       )


        #output = tf.sparse_tensor_dense_matmul(h,weights2)
        lookup2 = tf.nn.embedding_lookup(weights2, positive_idx(h))

        #seg_prod = tf.segment_prod(weights2,h_idx)
        #output = tf.reduce_sum(lookup2, axis=1)
        #output1 = tf.matmul(lookup2,h_v)
        #output2 = tf.reduce_sum(output1,axis=1)




        # create a random input to simulate a RI sample
        gen = Generator(active=n_active, dim=dim)
        ri1 = gen.generate()
        ri2 = gen.generate()
        ri_indexes_1 = ri1.positive + ri1.negative
        ri_indexes_vec_1 = ri_indexes_1
        ri_indexes_1 = np.asmatrix(ri_indexes_1)

        ri_vector_1 = np.zeros(dim)
        ri_vector_1[ri_indexes_1] = 1
        ri_vector_1 = np.asmatrix(ri_vector_1)

        ri_indexes_2 = ri2.positive + ri2.negative
        ri_indexes_vec_2 = ri_indexes_2
        ri_indexes_2 = np.asmatrix(ri_indexes_2)

        ri_vector_2 = np.zeros(dim)
        ri_vector_2[ri_indexes_2] = 1
        ri_vector_2 = np.asmatrix(ri_vector_2)



        target = tf.placeholder(shape=[1, dim], dtype=tf.float32, name="labels")
        loss = tf.losses.mean_squared_error(labels=target, predictions=output)


        var_grad = tf.gradients(loss, weights2)


        init = tf.global_variables_initializer()
        with tf.Session() as ss:
            ss.run(init)

            #result = ss.run(output1, feed_dict={id_input: ri_indexes})

            #print(result)

            relu_act = ss.run(h,feed_dict={id_input: [ri_indexes_vec_1,ri_indexes_vec_2]})
#            relu_sparse_act = ss.run(h_s,feed_dict={id_input: ri_indexes})

            print("ReLU activation")
            for act in relu_act:
                print(act)
          #  print("ReLU sparse activation", relu_sparse_act)
            #print("sparse ReLU activation", ss.run(sparse_h, feed_dict={id_input: [ri_indexes_vec,ri_indexes_vec]}))


            print(ss.run(dense_to_sparse_ids(h), feed_dict={id_input: [ri_indexes_vec_1,ri_indexes_vec_2]}))
            print("\nvalues")
            print(ss.run(dense_to_sparse_values(h), feed_dict={id_input: [ri_indexes_vec_1,ri_indexes_vec_2]}))


            print("\n")
            print("Weights 2:", ss.run(weights2))
            #print("Sparse Weights", ss.run(h_v, feed_dict={id_input: ri_indexes}))
            #print("Sparse Ids", ss.run(tf.reshape(tf.where(condition=tf.not_equal(h, 0),),[-1,2]), feed_dict={id_input: [ri_indexes_vec,ri_indexes_vec]}))


            #print("Sparse lookup", ss.run(lookup2, feed_dict={id_input: ri_indexes}))

            print("\n")

            #print("Weighted lookup", ss.run(output1, feed_dict={id_input: ri_indexes}))
            print("Sparse Result lookup:")

            print(ss.run(tf.nn.embedding_lookup_sparse(params=weights2,
                                                       sp_ids=dense_to_sparse_ids(h),
                                                       sp_weights=None,
                                                       combiner="sum"
                                                       #sp_weights=dense_to_sparse_values(h)
                                                       ), feed_dict={id_input: [ri_indexes_vec_1]}))

            rs_t = tf.reshape(h, [-1])
            test_ids = tf.mod(tf.squeeze(tf.where(tf.not_equal(rs_t, 0))), h_dim)

            print(ss.run(test_ids,feed_dict={id_input: [ri_indexes_vec_1]}))
            test_lookup = tf.nn.embedding_lookup(weights2,[test_ids])


            print(ss.run(tf.reduce_sum(test_lookup, axis=1), feed_dict={id_input: [ri_indexes_vec_1]}))


            result_grad = ss.run(var_grad, feed_dict={id_input: [ri_indexes_vec_1], target: ri_vector_1})
            print(result_grad)