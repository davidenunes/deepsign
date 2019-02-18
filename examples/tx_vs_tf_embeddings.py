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

embed = tf.keras.layers.Embedding(inputs.n_units, embed_size)
embed_out = embed(inputs.tensor)
embed.embeddings = lookup.weights
embed_out = embed(inputs.tensor)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tx_weights = lookup.weights.eval()
    tf_weights = embed.embeddings.eval()

    np.testing.assert_array_equal(tx_weights, tf_weights)

    tx_out = lookup.eval()
    tf_out = embed_out.eval()

    np.testing.assert_array_equal(tx_out, tf_out)
