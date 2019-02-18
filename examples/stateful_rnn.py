from tensorx import *
import tensorflow as tf
import numpy as np

n_features = 5
embed_size = 4
hdim = 3
seq_size = 3
batch_size = 2

inputs = TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
lookup = Lookup(inputs, seq_size=seq_size, lookup_shape=[n_features, embed_size])
seq = lookup.permute_batch_time()

# this state is passed to the first cell instance which
# which transforms it into a list, the recurrent cell gets that state back as list
ones_state = tf.ones([batch_size, hdim])
zero_state = (tf.zeros([batch_size, hdim]))


def rnn_proto(x, **kwargs): return RNNCell(x, n_units=hdim, **kwargs)


rnn1 = RNN(seq, cell_proto=rnn_proto, previous_state=ones_state)