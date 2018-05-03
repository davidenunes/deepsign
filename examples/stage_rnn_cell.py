import tensorflow as tf
import tensorx as tx
from tensorx.layers import layer_scope, Layer
import numpy as np


class RNNCell(Layer):
    def __init__(self, layer, n_units, previous_output=None, activation=tx.tanh, use_bias=True,
                 init=tx.xavier_init(),
                 recurrent_init=tx.xavier_init(),
                 name="lstm_cell"):
        self.activation = activation
        self.use_bias = use_bias
        self.init = init

        self.recurrent_init = recurrent_init
        super().__init__(layer, n_units, [layer.n_units, n_units], tf.float32, name)

        self.tensor = self._build_graph(layer, previous_output)

    def _build_graph(self, layer, previous_output):
        with layer_scope(self, var_scope=True):
            self.previous_output = previous_output
            if self.previous_output is None:
                input_batch = tf.shape(layer.tensor)[0]
                # print(input_batch)
                self.previous_output = tf.zeros([input_batch, self.n_units])

            self.kernel = tx.Linear(layer, self.n_units, bias=True, init=self.init)
            self.kernel = tx.Activation(self.kernel, self.activation)

            self.recurrent_kernel = tx.Linear(tx.TensorLayer(self.previous_output, self.n_units),
                                              self.n_units,
                                              bias=False, init=self.recurrent_init)

            return self.kernel.tensor + self.recurrent_kernel.tensor


"""
Test staged implementation
"""
n_hidden = 20
embed_dim = 10
seq_size = 2
vocab_size = 10000
feature_shape = [vocab_size, embed_dim]

loss_inputs = tx.Input(1, dtype=tf.int32)
in_layer = tx.Input(seq_size, dtype=tf.int32)

lookup = tx.Lookup(in_layer, seq_size=seq_size,
                   feature_shape=feature_shape)
# [batch x seq_size * feature_shape[1]]

# reshape to [batch x seq_size x feature_shape[1]]
# lookup_to_seq =
out = tx.WrapLayer(lookup, embed_dim, shape=[None, seq_size, embed_dim],
                   tf_fn=lambda tensor: tf.reshape(tensor, [-1, seq_size, embed_dim]))

out = tx.WrapLayer(out, embed_dim, tf_fn=lambda tensor: tensor[0])
# apply rnn cell to single input batch

out_rnn = RNNCell(out, 4)

# setup optimizer
optimizer = tx.AMSGrad(learning_rate=0.01)

model = tx.Model(run_in_layers=in_layer, run_out_layers=out_rnn)
runner = tx.ModelRunner(model)
runner.init_vars()

# need to fix the runner interface to allow for lists to be received
data = np.array([[0, 1], [1, 0]])
targets = np.array([[2], [3]])

result = runner.run(data)
print(result)

# should be something like
#
# rnn = RNN(RNNCell(input_layer,num_hidden), steps=3)
# rnn.tensor  has shape [batch, steps,num_hidden]
# I can add indexing for rnn returning an RNNCell
