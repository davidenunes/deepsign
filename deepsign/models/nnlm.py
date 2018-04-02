import tensorx as tx
import tensorflow as tf


class NNLM(tx.Model):
    """ Neural Probabilistic Language Model


    if use_f_predict is True, this model can be interpreted as an

    Energy-based Neural Network Language Modelling network

    Same as Bengio Neural Probabilistic Language Model but with a linear layer
    at the end feature_pred with the same dimensions as the embeddings and possibly
    with embedding sharing between input and output layers

    """

    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.elu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 embed_dropout=False,
                 keep_prob=0.95,
                 l2_loss=False,
                 l2_loss_coef=1e-5,
                 use_f_predict=False,
                 f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=False
                 ):

        run_inputs = tx.Input(ctx_size, dtype=tf.int32)
        loss_inputs = tx.Input(n_units=1, dtype=tf.int64)
        eval_inputs = loss_inputs

        if run_inputs.dtype != tf.int32 and run_inputs.dtype != tf.int64:
            raise TypeError("Invalid dtype for input: expected int32 or int64, got {}".format(run_inputs.dtype))

        if num_h < 0:
            raise ValueError("num hidden should be >= 0")

        # ===============================================
        # RUN GRAPH
        # ===============================================
        var_reg = []

        with tf.name_scope("run"):
            # feature lookup
            feature_lookup = tx.Lookup(run_inputs, ctx_size, [vocab_size, embed_dim], weight_init=embed_init)
            var_reg.append(feature_lookup.weights)

            last_layer = feature_lookup
            h_layers = []
            for i in range(num_h):
                h_i = tx.Linear(last_layer, h_dim, h_init, bias=True, name="h_{i}_linear".format(i=i))
                h_a = tx.Activation(h_i, h_activation)
                h = tx.Compose([h_i, h_a], name="h_{i}".format(i=i))
                h_layers.append(h)
                last_layer = h
                var_reg.append(h_i.weights)

            # feature prediction for Energy-Based Model
            if use_f_predict:
                last_layer = tx.Linear(last_layer, embed_dim, f_init, "f_predict")
                var_reg.append(last_layer.weights)

            shared_weights = tf.transpose(feature_lookup.weights) if embed_share else None
            logit_init = logit_init if not embed_share else None
            run_logits = tx.Linear(last_layer, vocab_size, logit_init, shared_weights, name="logits")

            if not embed_share:
                var_reg.append(run_logits.weights)

            run_output = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================
        with tf.name_scope("train"):
            if use_dropout and embed_dropout:
                last_layer = tx.Dropout(feature_lookup, keep_prob=keep_prob)
            else:
                last_layer = feature_lookup

            # add dropout between each layer
            for layer in h_layers:
                h = layer.reuse_with(last_layer)
                if use_dropout:
                    h = tx.Dropout(h, keep_prob=keep_prob)
                last_layer = h

            train_logits = run_logits.reuse_with(last_layer, name="train_logits")
            train_output = tx.Activation(train_logits, tx.softmax, name="train_output")

            one_hot = tx.dense_one_hot(column_indices=loss_inputs.tensor, num_cols=vocab_size)
            train_loss = tx.categorical_cross_entropy(one_hot, train_logits.tensor)

            train_loss = tf.reduce_mean(train_loss)

            if l2_loss:
                losses = [tf.nn.l2_loss(var) for var in var_reg]
                train_loss = train_loss + l2_loss_coef * tf.add_n(losses)

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        with tf.name_scope("eval"):
            one_hot = tx.dense_one_hot(column_indices=eval_inputs.tensor, num_cols=vocab_size)
            eval_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)
            eval_loss = tf.reduce_mean(eval_loss)

        # BUILD MODEL
        super().__init__(run_in_layers=run_inputs, run_out_layers=run_output,
                         train_in_layers=run_inputs, train_out_layers=train_output,
                         eval_in_layers=run_inputs, eval_out_layers=run_output,
                         train_loss_tensors=train_loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)
