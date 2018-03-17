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
                 feature_dim,
                 feature_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 y_pred_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 feature_share=True,
                 nce=False,
                 nce_samples=100):
        run_inputs = tx.Input(ctx_size, dtype=tf.int32)
        loss_inputs = tx.Input(n_units=1, dtype=tf.int32)
        eval_inputs = loss_inputs

        # RUN GRAPH
        lookup = tx.Lookup(run_inputs, ctx_size, [vocab_size, feature_dim], feature_init)
        y_pred = tx.Linear(lookup, feature_dim, y_pred_init)

        shared_weights = tf.transpose(lookup.weights) if feature_share else None
        logit_init = logit_init if not feature_share else None
        logits = tx.Linear(y_pred, vocab_size, logit_init, shared_weights)
        y_prob = tx.Activation(logits, tx.softmax)

        # TRAIN GRAPH ===============================================
        if nce:
            # uniform gets good enough results if enough samples are used
            # but we can load the empirical unigram distribution
            # or learn the unigram distribution during training
            sampled_values = uniform_sampler(loss_inputs.tensor, 1, nce_samples, True, vocab_size)
            loss = tf.nn.nce_loss(weights=tf.transpose(logits.weights),
                                  biases=logits.bias,
                                  inputs=y_pred.tensor,
                                  labels=loss_inputs.tensor,
                                  num_sampled=nce_samples,
                                  num_classes=vocab_size,
                                  num_true=1,
                                  sampled_values=sampled_values)
        else:
            one_hot = tx.dense_one_hot(column_indices=loss_inputs.tensor, num_cols=vocab_size)
            loss = tx.categorical_cross_entropy(one_hot, logits.tensor)
        loss = tf.reduce_mean(loss)

        # EVAL GRAPH ===============================================
        one_hot = tx.dense_one_hot(column_indices=eval_inputs.tensor, num_cols=vocab_size)
        eval_loss = tx.categorical_cross_entropy(one_hot, logits.tensor)
        eval_loss = tf.reduce_mean(eval_loss)

        # SETUP MODEL CONTAINER ====================================
        super().__init__(run_in_layers=run_inputs, run_out_layers=y_prob,
                         train_in_layers=run_inputs, train_out_layers=y_prob,
                         eval_in_layers=run_inputs, eval_out_layers=y_prob,
                         train_loss_tensors=loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)
