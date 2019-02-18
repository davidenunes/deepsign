import tensorflow as tf
import tensorx as tx
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

n_features = 3
embed_size = 4
cell_units = 2
seq_size = 3
batch_size = 2

inputs = tx.TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
lookup = tx.Lookup(inputs, seq_size=seq_size, lookup_shape=[n_features, embed_size])
seq = lookup.permute_batch_time()

# first step of a sequence
t1 = seq[0]


ks_cell = tf.keras.layers.LSTMCell(units=cell_units)
tf_cell = tf.nn.rnn_cell.LSTMCell(num_units=cell_units, state_is_tuple=True)
tx_cell = tx.LSTMCell(t1, n_units=cell_units)

kernel_w = [
    tx_cell.w_i.weights,
    tx_cell.w_c.weights,
    tx_cell.w_f.weights,
    tx_cell.w_o.weights]
kernel_u = [
    tx_cell.u_i.weights,
    tx_cell.u_c.weights,
    tx_cell.u_f.weights,
    tx_cell.u_o.weights]

# [input_depth + h_depth, 4 * self._num_units],
kernel_w = tx.Concat(*kernel_w)
kernel_u = tx.Concat(*kernel_u)
tx_kernel = tx.Merge(kernel_w, kernel_u, merge_fn=lambda l: tf.concat(l, axis=0))

# kernel = tx.Reshape(kernel, [-1, 4 * cell_units])

tf_zero_state = tf_cell.zero_state(batch_size, dtype=tf.float32)
tf_out, tf_state = tf_cell(t1.tensor, state=tf_zero_state)

# inject my internal state into TensorFlow lstm
tf_cell._kernel = tx_kernel
tf_out, tf_state = tf_cell(t1.tensor, state=tf_zero_state)

tx_rnn = tx.RNN(seq,
                cell_proto=lambda x, **kwargs: tx_cell.reuse_with(x, **kwargs),
                stateful=False)
tx_rnn = tx.Transpose(tx_rnn, [1, 0, 2])

# time major maintains the format in the output
# if time major output is time major
# if batch major, output is batch major
tf_rnn, tf_state = tf.nn.dynamic_rnn(
    cell=tf_cell,
    inputs=lookup.tensor,
    sequence_length=None,
    initial_state=tf_zero_state,
    time_major=False,
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    out1 = sess.run(tf_out)
    out2 = sess.run(tx_cell.tensor)

    np.testing.assert_array_almost_equal(out1, out2, decimal=7)

    tx_rnn_out = sess.run(tx_rnn.tensor)
    tf_rnn_out = sess.run(tf_rnn)

    np.testing.assert_array_almost_equal(tx_rnn_out, tf_rnn_out, decimal=7)
    print("tx RNN \n", tx_rnn_out)
    print("tf RNN \n", tf_rnn_out)
