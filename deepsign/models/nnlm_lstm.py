import tensorx as tx
import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler
import sys
from functools import partial


class LSTM_NNLM(tx.Model):
    """ Neural Probabilistic Language Model


    if use_f_predict is True, this model can be interpreted as an

    Energy-based Neural Network Language Modelling network

    Same as Bengio Neural Probabilistic Language Model but with a linear layer
    at the end feature_pred with the same dimensions as the embeddings and possibly
    with embedding sharing between input and output layers

    """

    def __init__(self,
                 inputs,
                 labels,
                 vocab_size,
                 embed_dim,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.tanh,
                 h_init=tx.random_uniform(minval=-0.05, maxval=0.05),
                 reset_state=True,
                 embed_dropout=False,
                 w_dropout=False,
                 u_dropconnect=False,
                 other_dropout=False,
                 w_keep_prob=0.9,
                 u_keep_prob=0.9,
                 embed_keep_prob=0.9,
                 other_keep_prob=0.9,
                 l2_loss=False,
                 l2_weight=1e-5,
                 use_f_predict=False,
                 f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=False,
                 logit_bias=False,
                 use_nce=False,
                 nce_samples=10,
                 ):
        if not isinstance(inputs, tx.Input):
            raise TypeError("inputs must be an Input layer")
        self.inputs = inputs
        self.labels = labels
        if not isinstance(labels, tx.Input):
            raise TypeError("labels must be an Input layer")

        if inputs.dtype != tf.int32 and inputs.dtype != tf.int64:
            raise TypeError("Invalid dtype for input: expected int32 or int64, got {}".format(inputs.dtype))

        if num_h < 0:
            raise ValueError("num hidden should be >= 0")

        # ===============================================
        # RUN GRAPH
        # ===============================================
        var_reg = []

        with tf.name_scope("run"):
            # feature lookup
            embeddings = tx.Lookup(inputs,
                                   seq_size=None,
                                   lookup_shape=[vocab_size, embed_dim],
                                   weight_init=embed_init)
            var_reg.append(embeddings.weights)
            feature_lookup = embeddings.as_seq()

            last_layer = feature_lookup

            w_reg = partial(tx.Dropout, keep_prob=w_keep_prob) if w_dropout else None
            u_reg = partial(tx.DropConnect, keep_prob=u_keep_prob) if u_dropconnect else None

            def cell_proto(x, **kwargs):
                return tx.LSTMCell(x,
                                   n_units=h_dim,
                                   activation=h_activation,
                                   w_init=h_init,
                                   u_init=h_init,
                                   w_regularizer=w_reg,
                                   u_regularizer=u_reg,
                                   regularized=False,
                                   name="lstm_cell",
                                   **kwargs)

            lstm_layers = []
            for i in range(num_h):
                lstm_layer = tx.Recurrent(last_layer,
                                          cell_proto=cell_proto,
                                          regularized=False,
                                          stateful=True,
                                          name="LSTM_{}".format(i+1))

                lstm_layers.append(lstm_layer)
                var_reg += [wi.weights for wi in lstm_layer.cell.w]
                var_reg += [ui.weights for ui in lstm_layer.cell.u]
                last_layer = lstm_layer

            # last time step is the state used to make the prediction
            last_layer = last_layer[-1]

            # feature prediction for Energy-Based Model
            if use_f_predict:
                last_layer = tx.Linear(last_layer, embed_dim, f_init, add_bias=True, name="f_predict")
                var_reg.append(last_layer.weights)
                f_predict = last_layer

            shared_weights = feature_lookup.weights if embed_share else None
            transpose_weights = embed_share
            logit_init = logit_init if not embed_share else None
            run_logits = tx.Linear(last_layer,
                                   n_units=vocab_size,
                                   weight_init=logit_init,
                                   shared_weights=shared_weights,
                                   transpose_weights=transpose_weights,
                                   add_bias=logit_bias,
                                   name="logits")

            if not embed_share:
                var_reg.append(run_logits.weights)

            run_output = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================
        with tf.name_scope("train"):
            embeddings = embeddings.reuse_with(inputs)
            feature_lookup = embeddings.as_seq()

            if other_dropout and embed_dropout:
                feature_lookup = tx.Dropout(feature_lookup, keep_prob=embed_keep_prob, name="drop_features")

            last_layer = feature_lookup

            for i in range(num_h):
                lstm_layer = lstm_layers[i].reuse_with(last_layer, regularized=True)
                last_layer = lstm_layer

            last_layer = last_layer[-1]

            # feature prediction for Energy-Based Model
            if use_f_predict:
                last_layer = f_predict.reuse_with(last_layer)

            train_logits = run_logits.reuse_with(last_layer, name="train_logits")
            train_output = tx.Activation(train_logits, tx.softmax, name="train_output")

            def categorical_loss(labels, logits):
                labels = tx.dense_one_hot(column_indices=labels, num_cols=vocab_size)
                loss = tx.categorical_cross_entropy(labels=labels, logits=logits)
                # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits)
                return tf.reduce_mean(loss)

            def nce_loss(labels, weights, bias, predict):
                noise = uniform_sampler(labels, 1, nce_samples, True, vocab_size)
                loss = tf.nn.nce_loss(weights=weights,
                                      biases=bias,
                                      inputs=predict,
                                      labels=labels,
                                      num_sampled=nce_samples,
                                      num_classes=vocab_size,
                                      num_true=1,
                                      sampled_values=noise)
                return tf.reduce_mean(loss)

            if use_nce:
                bias = tx.VariableLayer(var_shape=[vocab_size], name="nce_bias")

                nce_weights = tx.WrapLayer(embeddings,
                                           n_units=embeddings.n_units,
                                           wrap_fn=lambda x: x.weights,
                                           layer_fn=True)
                train_loss = tx.FnLayer(labels, nce_weights, bias, last_layer, apply_fn=nce_loss, name="nce_loss")
            else:
                train_loss = tx.FnLayer(labels, train_logits, apply_fn=categorical_loss, name="train_loss")

            if l2_loss:
                l2_losses = [tf.nn.l2_loss(var) for var in var_reg]
                train_loss = tx.FnLayer(train_loss,
                                        apply_fn=lambda x: x + l2_weight * tf.add_n(l2_losses),
                                        name="train_loss_l2")

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        with tf.name_scope("eval"):
            eval_loss = tx.FnLayer(labels, run_logits, apply_fn=categorical_loss, name="eval_loss")

        self.stateful_layers = lstm_layers
        # BUILD MODEL
        super().__init__(run_outputs=run_output,
                         run_inputs=inputs,
                         train_inputs=[inputs, labels],
                         train_outputs=train_output,
                         train_loss=train_loss,
                         eval_inputs=[inputs, labels],
                         eval_outputs=run_output,
                         eval_score=eval_loss)

    def reset_state(self):
        """ returns an op that resets all stateful layers

        """
        reset_op = tf.group(*[layer.reset() for layer in self.stateful_layers])
        self.session.run(reset_op)
