import tensorflow as tf
import tensorx as tx
import numpy as np

ss = tf.InteractiveSession()

v_dim = 1000
m_dim = 2
n_hidden = 100
seq_size = 2

w = [[0, 1], [1, 5], [0, 1]]
v2 = tf.constant(np.random.uniform(-1., 1., [v_dim, m_dim]))

inputs = tx.Input(2, dtype=tf.int32)

lookup = tx.Lookup(inputs, 2, lookup_shape=[v_dim, m_dim])

# GATING MECHANISM
# I can call this a seq gate, takes the parameters and divides by seq_size
h = tx.Linear(lookup, 100, bias=True)
h = tx.Activation(h, tx.elu)

gate = tx.Linear(h, 2, bias=True)
gate = tx.Activation(gate, tx.sigmoid)

# lookup might output a sequence format with [batch,seq_size,m_dim]
# lookup_out = lookup.tensor
lookup_out = tf.reshape(lookup.tensor, [-1, seq_size, m_dim])

# reshape works anyway
gated_out = tf.reshape(lookup_out, [-1, seq_size, m_dim]) * tf.expand_dims(gate.tensor, -1)

# gated_out = tf.reshape(gated_out, [-1, seq_size * m_dim])
# gated_out = tf.reshape(gated_out, [-1, lookup.n_units])
gated_out = tf.reshape(gated_out, tf.shape(lookup.tensor))
gated_out = tx.TensorLayer(gated_out, lookup.n_units)
# END GATING MECHANISM

y = tx.Linear(gated_out, m_dim, bias=True)

ss.run(tf.global_variables_initializer())

lookup_out = lookup.tensor.eval({inputs.placeholder: w})

assert (np.shape(lookup_out) == (3, 2 * m_dim))
print(np.shape(lookup_out))

gated_out = gated_out.tensor.eval({inputs.placeholder: w})
print(np.shape(gated_out))

gate_values = gate.tensor.eval({inputs.placeholder: w})

print("lookup out")
print(lookup_out)
print("gates: {}".format(str(gate_values)))
print("lookup gated")
print(gated_out)
