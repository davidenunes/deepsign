""" Log Bi-Linear Language Model

"""
import tensorx as tx
import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler


class LBL(tx.Model):
    """
    Args:
            ctx_size:
            feature_dim: feature / embedding dimension
            feature_share: share the features with target words
            nce: bool if true uses NCE approximation to softmax
            nce_samples: number of samples for nce
    """

    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 y_pred_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=True,
                 use_gate=True,
                 use_hidden=False,
                 h_dim=100,
                 h_activation=tx.elu,
                 h_init=tx.he_normal_init(),
                 w_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 use_dropout=True,
                 embed_dropout=False,
                 keep_prob=0.95,
                 l2_loss=False,
                 l2_loss_coef=1e-5,
                 nce=False,
                 nce_samples=100):
        run_inputs = tx.Input(ctx_size, dtype=tf.int32)
        loss_inputs = tx.Input(n_units=1, dtype=tf.int32)
        eval_inputs = loss_inputs

        # RUN GRAPH
        lookup = tx.Lookup(run_inputs, ctx_size, [vocab_size, embed_dim], embed_init, name="embed")

        if use_gate or use_hidden:
            hl = tx.Linear(lookup, h_dim, h_init, name="h_wb")
            ha = tx.Activation(hl, h_activation, name="h_fn")
            h = tx.Compose([hl, ha], name="h")

        features = lookup
        if use_gate:
            features = tx.Gate(features, ctx_size, gate_input=h)
            gate = features

        x_to_y = tx.Linear(features, embed_dim, y_pred_init, name="x_y")
        y_pred = x_to_y
        if use_hidden:
            h_to_y = tx.Linear(h, embed_dim, w_init, name="h_y")
            y_pred = tx.Add([y_pred, h_to_y])

        shared_weights = tf.transpose(lookup.weights) if embed_share else None
        logit_init = logit_init if not embed_share else None
        run_logits = tx.Linear(y_pred, vocab_size, logit_init, shared_weights, name="logits")
        y_prob = tx.Activation(run_logits, tx.softmax)

        # TRAIN GRAPH ===============================================
        if use_dropout and embed_dropout:
            features = tx.Dropout(lookup, keep_prob=keep_prob)
        else:
            features = lookup

        if l2_loss:
            weights_losses = [tf.nn.l2_loss(lookup.weights)]

        if use_gate or use_hidden:
            h = h.reuse_with(features)

            if l2_loss:
                weights_losses.append(tf.nn.l2_loss(h.layers[0].weights))
                weights_losses.append(tf.nn.l2_loss(y_pred.weights))

            if use_gate:
                features = gate.reuse_with(features)
                if l2_loss:
                    weights_losses.append(tf.nn.l2_loss(gate.gate.layers[0].weights))

            y_pred = x_to_y.reuse_with(features)

            if use_hidden:
                h_to_y = h_to_y.reuse_with(h)
                if l2_loss:
                    weights_losses.append(tf.nn.l2_loss(h_to_y.weights))
                y_pred = tx.Add([y_pred, h_to_y])
        else:
            y_pred = y_pred.reuse_with(features)
            if l2_loss:
                weights_losses.append(tf.nn.l2_loss(y_pred.weights))

        train_logits = run_logits.reuse_with(y_pred)

        if l2_loss and not embed_share:
            weights_losses.append(tf.nn.l2_loss(train_logits.weights))

        if nce:
            # uniform gets good enough results if enough samples are used
            # but we can load the empirical unigram distribution
            # or learn the unigram distribution during training
            sampled_values = uniform_sampler(loss_inputs.tensor, 1, nce_samples, True, vocab_size)
            train_loss = tf.nn.nce_loss(weights=tf.transpose(train_logits.weights),
                                        biases=train_logits.bias,
                                        inputs=y_pred.tensor,
                                        labels=loss_inputs.tensor,
                                        num_sampled=nce_samples,
                                        num_classes=vocab_size,
                                        num_true=1,
                                        sampled_values=sampled_values)
        else:
            one_hot = tx.dense_one_hot(column_indices=loss_inputs.tensor, num_cols=vocab_size)
            train_loss = tx.categorical_cross_entropy(one_hot, train_logits.tensor)

        train_loss = tf.reduce_mean(train_loss)

        if l2_loss:
            train_loss = train_loss + l2_loss_coef * tf.add_n(weights_losses)

        # EVAL GRAPH ===============================================
        one_hot = tx.dense_one_hot(column_indices=eval_inputs.tensor, num_cols=vocab_size)
        eval_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)
        eval_loss = tf.reduce_mean(eval_loss)

        # SETUP MODEL CONTAINER ====================================
        super().__init__(run_in_layers=run_inputs, run_out_layers=y_prob,
                         train_in_layers=run_inputs, train_out_layers=y_prob,
                         eval_in_layers=run_inputs, eval_out_layers=y_prob,
                         train_loss_tensors=train_loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)
