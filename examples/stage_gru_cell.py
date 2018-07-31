import tensorflow as tf
import tensorx as tx
from tensorx.layers import layer_scope, Layer
import numpy as np


class GRUCell(Layer):
    """ Gated Recurrent Cell
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

            https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1426
            https://en.wikipedia.org/wiki/Gated_recurrent_unit
            https://www.coursera.org/learn/nlp-sequence-models/lecture/agZiL/gated-recurrent-unit-gru
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
                # determines the weight of the previous state
                # we could add the bias at the end but this way we just define a single bias for the r unit
                self.r_current_w = tx.Linear(layer, self.n_units, bias=True, init=self.init, name="r_current_w")
                self.r_recurrent_w = tx.Linear(self.previous_state, self.n_units, bias=False, init=self.recurrent_init,
                                               name="r_current_w")

                self.u_current_w = tx.Linear(layer, self.n_units, bias=True, init=self.init, name="u_current_w")
                self.u_recurrent_w = tx.Linear(self.previous_state, self.n_units, bias=False, init=self.recurrent_init,
                                               name="u_current_w")

                self.current_w = tx.Linear(layer, self.n_units, bias=True, init=self.init, name="current_w")
                self.recurrent_w = tx.Linear(self.previous_state, self.n_units, bias=False, init=self.recurrent_init,
                                             name="recurrent_w")

                # kernel_gate = tx.Activation()

                kernel_act = tx.Activation(kernel_linear, self.activation)
                self.kernel = tx.Compose(kernel_linear, kernel_act)


            else:
                self.kernel = self.share_state_with.kernel.reuse_with(layer)
                self.recurrent_kernel = self.share_state_with.recurrent_kernel.reuse_with(self.previous_state)

            r_state = tx.Add(r_current_w, r_recurrent_w)
            r_state = tx.Bias(r_state)
            r_gate = tx.Activation(r_state, fn=tx.sigmoid, name="r_gate")

            # """Gated recurrent unit (GRU) with nunits cells."""
            return self.kernel.tensor + self.recurrent_kernel.tensor

    def reuse_with(self, layer, state=None, name=None):
        if state is None:
            state = self.previous_state

        if name is None:
            name = self.name

        return GRUCell(
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
embed_dim = 10
seq_size = 2
vocab_size = 10000
feature_shape = [vocab_size, embed_dim]

loss_inputs = tx.Input(1, dtype=tf.int32)
in_layer = tx.Input(seq_size, dtype=tf.int32)

lookup = tx.Lookup(in_layer, seq_size=seq_size,
                   lookup_shape=feature_shape)
# [batch x seq_size * feature_shape[1]]

# reshape to [batch x seq_size x feature_shape[1]]
# lookup_to_seq =
# I was thinking that this reshape could be done automatically based on the input share of
# the tensor fed to the RNN cell layer
out = tx.WrapLayer(lookup, embed_dim, shape=[None, seq_size, embed_dim],
                   tf_fn=lambda tensor: tf.reshape(tensor, [-1, seq_size, embed_dim]))

out = tx.WrapLayer(out, embed_dim, tf_fn=lambda tensor: tensor[0])
# apply rnn cell to single input batch

with tf.name_scope("rnn"):
    rnn1 = RNNCell(out, 4, name="rnn1")
    rnn2 = rnn1.reuse_with(out, state=rnn1, name="rnn2")
    rnn3 = rnn1.reuse_with(out, state=rnn2, name="rnn3")

# setup optimizer
optimizer = tx.AMSGrad(learning_rate=0.01)

model = tx.Model(run_in_layers=in_layer, run_out_layers=[rnn1, rnn2, rnn3])
runner = tx.ModelRunner(model)

runner.set_session(runtime_stats=True)
runner.log_graph(logdir="/tmp")
print("graph written")

runner.init_vars()

# need to fix the runner interface to allow for lists to be received
data = np.array([[0, 1], [1, 0]])
targets = np.array([[2], [3]])

result = runner.run(data)

for i, r in enumerate(result):
    print("rnn ", i)
    print(r)

# should be something like
#
# rnn = RNN(RNNCell(input_layer,num_hidden), steps=3)
# rnn.tensor  has shape [batch, steps,num_hidden]
# I can add indexing for rnn returning an RNNCell
