""" Log Bi-Linear Language Model

"""
import tensorx as tx
from tensorx.utils import to_tensor_cast
import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler
import numpy as np

from deepsign.models.ri_nce import ri_nce_loss, random_ri_nce_loss
from deepsign.rp.tf_utils import RandomIndexTensor


class LBL_NRP(tx.Model):
    """ LBL with NRP

    Log BiLinear Model with gating and hidden non-linear layer extensions using the random
    projection encoding

    Notes:
        - If embed_share is False, the model uses two embedding layers, one for the context words
        and one for the target words. The context word embeddings are initialized with the embed_init
        function. The target word embeddings are initialized with the logit_init function

    Args:
            ctx_size: number of words in the input context / sequence
            k_dim dimension for random projections, shoudl match dimension in ri_tensor
            ri_tensor: a sparse tensor with the sparse random indexes for each word, ordered by word index
            embed_dim: feature / embedding dimension
            embed_init: initialisation function for embedding weights
            embed_share: share the features with target words
            logit_init: initialisation function for logits only applicable if embed_share is set to False.
            h_to_f_init: if use hidden we have an additional set of parameters from the hidden layer to f_predict

    """

    def __init__(self,
                 ctx_size,
                 vocab_size,
                 k_dim,
                 ri_tensor: RandomIndexTensor,
                 embed_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 x_to_f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=True,
                 logit_biases=False,
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
                 l2_loss_coef=1e-5
                 ):

        # GRAPH INPUTS
        run_inputs = tx.Input(ctx_size, dtype=tf.int32, name="input")
        loss_inputs = tx.Input(n_units=1, dtype=tf.int32, name="target")
        eval_inputs = loss_inputs

        # RUN GRAPH =====================================================
        var_reg = []
        with tf.name_scope("run"):
            # RI ENCODING ===============================================
            # convert ids to ris gather a set of random indexes based on the ids in a sequence

            # ri_layer = tx.TensorLayer(ri_tensor, n_units=k_dim)
            # ri_inputs = tx.gather_sparse(ri_layer.tensor, run_inputs.tensor)
            with tf.name_scope("ri_encode"):
                # used to compute logits
                if isinstance(ri_tensor, RandomIndexTensor):
                    ri_layer = tx.TensorLayer(ri_tensor.to_sparse_tensor(), k_dim)

                    ri_inputs = ri_tensor.gather(run_inputs.tensor)
                    ri_inputs = ri_inputs.to_sparse_tensor()
                    ri_inputs = tx.TensorLayer(ri_inputs, k_dim)
                else:
                    ri_layer = tx.TensorLayer(ri_tensor, k_dim)
                    ri_inputs = tx.gather_sparse(ri_layer.tensor, run_inputs.tensor)
                    ri_inputs = tx.TensorLayer(ri_inputs, k_dim)

            # use those sparse indexes to lookup a set of features based on the ri values
            feature_lookup = tx.Lookup(ri_inputs, ctx_size, [k_dim, embed_dim], embed_init, name="lookup")
            var_reg.append(feature_lookup.weights)
            # ===========================================================

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
            shared_weights = feature_lookup.weights if embed_share else None
            logit_init = logit_init if not embed_share else None
            # embedding feature vectors for all words: shape [vocab_size, embed_dim]
            # later, for NCE we don't need to get all the features

            all_embeddings = tx.Linear(ri_layer, embed_dim, logit_init, shared_weights, name="logits", bias=False)

            # dot product of f_predicted . all_embeddings with bias for each target word

            run_logits = tx.Linear(f_prediction,
                                   n_units=vocab_size,
                                   shared_weights=all_embeddings.tensor,
                                   transpose_weights=True,
                                   bias=logit_biases)

            if not embed_share:
                var_reg.append(all_embeddings.weights)

            # ===========================================================
            run_embed_prob = tx.Activation(run_logits, tx.softmax)

        # TRAIN GRAPH ===================================================
        with tf.name_scope("train"):
            if use_dropout and embed_dropout:
                feature_lookup = feature_lookup.reuse_with(ri_inputs)
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

            # we already define all_embeddings from which these logits are computed before so this should be ok
            train_logits = run_logits.reuse_with(f_prediction)

            train_embed_prob = tx.Activation(train_logits, tx.softmax, name="train_output")

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
        super().__init__(run_in_layers=run_inputs, run_out_layers=run_embed_prob,
                         train_in_layers=run_inputs, train_out_layers=train_embed_prob,
                         eval_in_layers=run_inputs, eval_out_layers=run_embed_prob,
                         train_loss_tensors=train_loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)


class NNLM_NRP(tx.Model):
    """ Neural Probabilistic Language Model with NRP


    if use_f_predict is True, this model can be interpreted as an

    Energy-based Neural Network Language Modelling network

    Same as Bengio Neural Probabilistic Language Model but with a linear layer
    at the end feature_pred with the same dimensions as the embeddings and possibly
    with embedding sharing between input and output layers

    """

    def __init__(self,
                 ctx_size,
                 vocab_size,
                 k_dim,
                 s_active,
                 ri_tensor,
                 embed_dim,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.relu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 embed_dropout=False,
                 keep_prob=0.95,
                 l2_loss=False,
                 l2_loss_coef=1e-5,
                 f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 embed_share=False,
                 logit_biases=False,
                 use_nce=False,
                 nce_samples=100
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
            # RI ENCODING ===============================================
            # convert ids to ris gather a set of random indexes based on the ids in a sequence
            # ri_layer = tx.TensorLayer(ri_tensor, n_units=k_dim)
            # ri_inputs = tx.gather_sparse(ri_layer.tensor, run_inputs.tensor)
            # ri_inputs = tx.TensorLayer(ri_inputs, n_units=k_dim)
            with tf.name_scope("ri_encode"):
                # used to compute logits
                if isinstance(ri_tensor, RandomIndexTensor):
                    ri_layer = tx.TensorLayer(ri_tensor.to_sparse_tensor(), k_dim)

                    ri_inputs = ri_tensor.gather(run_inputs.tensor)
                    ri_inputs = ri_inputs.to_sparse_tensor()
                    ri_inputs = tx.TensorLayer(ri_inputs, k_dim)
                # ri_tensor is a sparse tensor
                else:
                    ri_layer = tx.TensorLayer(ri_tensor, k_dim)
                    ri_inputs = tx.gather_sparse(ri_layer.tensor, run_inputs.tensor)
                    ri_inputs = tx.TensorLayer(ri_inputs, k_dim)

            feature_lookup = tx.Lookup(ri_inputs, ctx_size, [k_dim, embed_dim], embed_init, name="lookup")
            var_reg.append(feature_lookup.weights)
            # ===========================================================

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

            f_prediction = tx.Linear(last_layer, embed_dim, f_init, name="f_predict")
            var_reg.append(f_prediction.weights)

            # RI DECODING ===============================================

            shared_weights = feature_lookup.weights if embed_share else None
            logit_init = logit_init if not embed_share else None

            ri_layer_dense = tx.ToDense(ri_layer)
            all_embeddings = tx.Linear(ri_layer_dense, embed_dim, logit_init, shared_weights, name="all_features",
                                       bias=False)

            # dot product of f_predicted . all_embeddings with bias for each target word
            run_logits = tx.Linear(f_prediction, vocab_size, shared_weights=all_embeddings.tensor,
                                   transpose_weights=True,
                                   bias=logit_biases, name="logits")

            if not embed_share:
                var_reg.append(all_embeddings.weights)
            # ===========================================================

            embed_prob = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================
        with tf.name_scope("train"):
            if use_dropout and embed_dropout:
                feature_lookup = feature_lookup.reuse_with(ri_inputs)
                last_layer = tx.Dropout(feature_lookup, keep_prob=keep_prob)
            else:
                last_layer = feature_lookup

            # add dropout between each layer
            for layer in h_layers:
                h = layer.reuse_with(last_layer)
                if use_dropout:
                    h = tx.Dropout(h, keep_prob=keep_prob)
                last_layer = h

            f_prediction = f_prediction.reuse_with(last_layer)

            train_logits = run_logits.reuse_with(f_prediction, name="train_logits")
            train_embed_prob = tx.Activation(train_logits, tx.softmax, name="train_output")

            if use_nce:
                train_loss = random_ri_nce_loss(ri_tensors=ri_layer.tensor, k_dim=k_dim, s_active=s_active,
                                                weights=feature_lookup.weights,
                                                labels=loss_inputs.tensor,
                                                inputs=f_prediction.tensor,
                                                num_sampled=nce_samples,
                                                num_classes=vocab_size,
                                                num_true=1,
                                                )
            else:
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
        super().__init__(run_in_layers=run_inputs, run_out_layers=embed_prob,
                         train_in_layers=run_inputs, train_out_layers=train_embed_prob,
                         eval_in_layers=run_inputs, eval_out_layers=embed_prob,
                         train_loss_tensors=train_loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs)
