import tensorflow as tf
import tensorx as tx
from tensorx.layers import layer_scope, Layer
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RNNCell(Layer):
    """ Recurrent Cell
    Corresponds to a single step on an unrolled RNN network

    Args:
            layer: the input layer to the RNN Cell
            n_units: number of output units for this RNN Cell
            previous_state: a RNNCell from which we can extract output
            activation: activation function to be used in the cell
            use_bias: if True adds biases before the activation
            init: weight initialisation function
            recurrent_init: initialisation function for the recurrent weights
            share_state_with: a ``Layer`` with the same number of units than this Cell
            name: name for the RNN cell
    """

    def __init__(self, layer, n_units,
                 previous_state=None,
                 activation=tx.tanh,
                 use_bias=True,
                 init=tx.xavier_init(),
                 recurrent_init=tx.xavier_init(),
                 share_state_with=None,
                 name="rnn_cell"):
        self.activation = activation
        self.use_bias = use_bias
        self.init = init

        self.recurrent_init = recurrent_init
        super().__init__(layer, n_units, [layer.n_units, n_units], tf.float32, name)

        if previous_state is not None:
            if previous_state.n_units != self.n_units:
                raise ValueError(
                    "previous state n_units ({}) != current n_units ({})".format(previous_state.n_units, self.n_units))
        self.previous_state = previous_state

        if share_state_with is not None and not isinstance(share_state_with, RNNCell):
            raise TypeError("shared_gate must be of type {} got {} instead".format(RNNCell, type(share_state_with)))
        self.share_state_with = share_state_with

        self.tensor = self._build_graph(layer, previous_state)

    def _build_graph(self, layer, previous_state):
        with layer_scope(self):

            if previous_state is None:
                input_batch = tf.shape(layer.tensor)[0]
                zero_state = tf.zeros([input_batch, self.n_units])
                self.previous_state = tx.TensorLayer(zero_state, self.n_units)

            if self.share_state_with is None:
                kernel_linear = tx.Linear(layer, self.n_units, bias=True, weight_init=self.init, name="linear_kernel")
                kernel_act = tx.Activation(kernel_linear, self.activation)
                self.kernel = tx.Compose([kernel_linear, kernel_act])

                self.recurrent_kernel = tx.Linear(self.previous_state,
                                                  self.n_units,
                                                  bias=False,
                                                  weight_init=self.recurrent_init,
                                                  name="recurrent_kernel")
            else:
                self.kernel = self.share_state_with.kernel.reuse_with(layer)
                self.recurrent_kernel = self.share_state_with.recurrent_kernel.reuse_with(self.previous_state)

            # TODO this might be wrong, I might need to couple the activation: act(kernel + recurrent + bias)
            # TODO it is wrong https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
            # """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
            return self.kernel.tensor + self.recurrent_kernel.tensor

    def reuse_with(self, layer, state=None, name=None):
        if state is None:
            state = self.previous_state

        if name is None:
            name = self.name

        return RNNCell(
            layer=layer,
            n_units=self.n_units,
            previous_state=state,
            activation=self.activation,
            use_bias=self.use_bias,
            share_state_with=self,
            name=name
        )


"""
Test staged implementation
"""
n_hidden = 20
embed_dim = 3
seq_size = 2
vocab_size = 10000
feature_shape = [vocab_size, embed_dim]

loss_inputs = tx.Input(1, dtype=tf.int32)
in_layer = tx.Input(seq_size, dtype=tf.int32)

lookup = tx.Lookup(in_layer,
                   seq_size=seq_size,
                   lookup_shape=feature_shape,
                   as_sequence=True)

lookup_flat = lookup.reuse_with(in_layer,as_sequence=False)


with tf.name_scope("rnn"):
    rnn1 = RNNCell(lookup[0], 4, name="rnn1")
    rnn2 = rnn1.reuse_with(lookup[1], state=rnn1, name="rnn2")


# setup optimizer
optimizer = tx.AMSGrad(learning_rate=0.01)

model = tx.Model(run_inputs=in_layer, run_outputs=[rnn1, rnn2])
runner = tx.ModelRunner(model)

runner.set_session(runtime_stats=True)
runner.log_graph(logdir="/tmp")
print("graph written")

runner.init_vars()

# need to fix the runner interface to allow for lists to be received
data = np.array([[1, 3], [1, 0]])
targets = np.array([[2], [3]])

flat_lookup = runner.session.run(lookup_flat.tensor, feed_dict={in_layer.placeholder: data})
seq_lookup = runner.session.run(lookup.tensor, feed_dict={in_layer.placeholder: data})
print("flat \n", flat_lookup)
print("seq \n", seq_lookup)


result = runner.run(data)

for i, r in enumerate(result):
    print("rnn ", i)
    print(r)

# should be something like
#
# rnn = RNN(RNNCell(input_layer,num_hidden), steps=3)
# rnn.tensor  has shape [batch, steps,num_hidden]
# I can add indexing for rnn returning an RNNCell
