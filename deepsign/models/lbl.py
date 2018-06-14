""" Log Bi-Linear Language Model

"""
import tensorx as tx
import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler


class LBL(tx.Model):
    """ Log BiLinear Model

    Log BiLinear Model with gating and hidden non-linear layer extensions.

    Notes:
        - If embed_share is False, the model uses two embedding layers, one for the context words
        and one for the target words. The context word embeddings are initialized with the embed_init
        function. The target word embeddings are initialized with the logit_init function

    Args:
            ctx_size: number of words in the input context / sequence
            embed_dim: feature / embedding dimension
            embed_init: initialisation function for embedding weights
            embed_share: share the features with target words
            nce: bool if true uses NCE approximation to softmax with random uniform samples
            nce_samples: number of samples for nce
            logit_init: initialisation function for logits only applicable if embed_share is set to False.
            h_to_f_init: if use hidden we have an additional set of parameters from the hidden layer to f_predict
    """

    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 x_to_f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=True,
                 use_gate=True,
                 use_hidden=False,
                 h_dim=100,
                 h_activation=tx.elu,
                 h_init=tx.he_normal_init(),
                 h_to_f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 use_dropout=True,
                 embed_dropout=False,
                 keep_prob=0.95,
                 l2_loss=False,
                 l2_loss_coef=1e-5,
                 use_nce=False,
                 nce_samples=100):

        # GRAPH INPUTS
        run_inputs = tx.Input(ctx_size, dtype=tf.int32, name="input")
        loss_inputs = tx.Input(n_units=1, dtype=tf.int32, name="target")
        eval_inputs = loss_inputs

        # RUN GRAPH
        # if I create a scope here the Tensorboard graph will be a mess to read
        # because it groups everything by nested scope names
        # instead if I choose to create different scopes for train and eval only
        # the graph stays readable because it allows us to use the same names
        # under different scopes while still sharing variables
        var_reg = []
        with tf.name_scope("run"):
            feature_lookup = tx.Lookup(run_inputs, ctx_size, [vocab_size, embed_dim], embed_init, name="lookup")
            var_reg.append(feature_lookup.weights)

            if use_gate or use_hidden:
                hl = tx.Linear(feature_lookup, h_dim, h_init, name="h_linear")
                ha = tx.Activation(hl, h_activation, name="h_activation")
                h = tx.Compose([hl, ha], name="hidden")
                var_reg.append(hl.weights)

            features = feature_lookup
            if use_gate:
                features = tx.Gate(features, ctx_size, gate_input=h)
                gate = features
                var_reg.append(features.gate_weights)

            x_to_f = tx.Linear(features, embed_dim, x_to_f_init, name="x_to_f")
            var_reg.append(x_to_f.weights)
            f_prediction = x_to_f

            if use_hidden:
                h_to_f = tx.Linear(h, embed_dim, h_to_f_init, name="h_to_f")
                var_reg.append(h_to_f.weights)
                f_prediction = tx.Add([x_to_f, h_to_f], name="f_predicted")

            # RI DECODING ===============================================
            shared_weights = tf.transpose(feature_lookup.weights) if embed_share else None
            logit_init = logit_init if not embed_share else None
            run_logits = tx.Linear(f_prediction, vocab_size, logit_init, shared_weights, name="logits")
            if not embed_share:
                var_reg.append(run_logits.weights)
            y_prob = tx.Activation(run_logits, tx.softmax)

        # TRAIN GRAPH ===============================================
        with tf.name_scope("train"):
            if use_dropout and embed_dropout:
                feature_lookup = feature_lookup.reuse_with(run_inputs)
                features = tx.Dropout(feature_lookup, keep_prob=keep_prob)
            else:
                features = feature_lookup

            if use_gate or use_hidden:
                if use_dropout:
                    h = h.reuse_with(features)
                    h = tx.Dropout(h, keep_prob=keep_prob)

                if use_gate:
                    features = gate.reuse_with(features, gate_input=h)

                f_prediction = x_to_f.reuse_with(features)

                if use_hidden:
                    h_to_f = h_to_f.reuse_with(h)
                    if use_dropout:
                        h_to_f = tx.Dropout(h_to_f, keep_prob=keep_prob)
                    f_prediction = tx.Add([f_prediction, h_to_f])
            else:
                f_prediction = f_prediction.reuse_with(features)

            train_logits = run_logits.reuse_with(f_prediction)

            if use_nce:
                # uniform gets good enough results if enough samples are used
                # but we can load the empirical unigram distribution
                # or learn the unigram distribution during training
                sampled_values = uniform_sampler(loss_inputs.tensor, 1, nce_samples, True, vocab_size)
                train_loss = tf.nn.nce_loss(weights=tf.transpose(train_logits.weights),
                                            biases=train_logits.bias,
                                            inputs=f_prediction.tensor,
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
                losses = [tf.nn.l2_loss(var) for var in var_reg]
                train_loss = train_loss + l2_loss_coef * tf.add_n(losses)

        # EVAL GRAPH ===============================================
        with tf.name_scope("eval"):
            one_hot = tx.dense_one_hot(column_indices=eval_inputs.tensor, num_cols=vocab_size)
            eval_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)
            eval_loss = tf.reduce_mean(eval_loss)

        # SETUP MODEL CONTAINER ====================================
        super().__init__(run_in_layers=run_inputs, run_out_layers=y_prob,
                         train_in_layers=run_inputs, train_out_layers=y_prob,
                         eval_in_layers=run_inputs, eval_out_layers=y_prob,
                         train_loss_tensors=train_loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)
