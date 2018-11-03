import tensorx as tx
import tensorflow as tf
from deepsign.rp.tf_utils import ris_to_sp_tensor_value
from deepsign.rp.index import SignIndex

from deepsign.rp.tf_utils import RandomIndexTensor


class NRP(tx.Model):
    """ Neural Probabilistic Language Model with NRP


    if use_f_predict is True, this model can be interpreted as an

    Energy-based Neural Network Language Modelling network

    Same as Bengio Neural Probabilistic Language Model but with a linear layer
    at the end feature_pred with the same dimensions as the embeddings and possibly
    with embedding sharing between input and output layers

    """

    def __init__(self,
                 ctx_size,
                 sign_index: SignIndex,
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
                 embed_share=True,
                 logit_bias=False,
                 use_nce=False,
                 nce_samples=100,
                 noise_level=0.1
                 ):

        self.embed_dim = embed_dim

        run_inputs = tx.SparseInput(k_dim, dtype=tf.int32)
        loss_inputs = tx.SparseInput(n_units=1, dtype=tf.int64)
        eval_inputs = loss_inputs

        var_reg = []

        # used for evaluation and inference with full normalisation
        # updated in self.update_state()
        # TODO this will be a dynamic var in the future with size [len(sign_index),embed_dim]
        # this might be a problem because inference expects a fixed size output layer
        # TODO unless we allow models to carry parameters that might be dynamic and fed through Params
        # like what we do with learning rates in ModelRunner
        with tf.name_scope("inference_evaluation"):
            self.all_embeddings = tf.get_variable("all_embeddings",
                                                  shape=[len(sign_index, embed_dim)],
                                                  dtype=tf.float32,
                                                  initializer=None,
                                                  trainable=False)

            ris = [sign_index.get_ri(sign_index.get_sign(i)) for i in range(len(sign_index))]
            self.all_ris = ris_to_sp_tensor_value(ri_seq=ris,
                                                  dim=sign_index.generator.dim,
                                                  all_positive=not sign_index.generator.symmetric)
            self.all_ris = tx.TensorLayer(tensor=self.all_ris,
                                          n_units=sign_index.feature_dim(),
                                          shape=[len(sign_index), sign_index.feature_dim()])

            """ returns a new op that updates the model state with all the variables after
                  train has been called
                  """
            all_embeddings = tx.Linear(self.all_ris,
                                       n_units=self.embed_dim,
                                       shared_weights=self.embedding_lookup.weights,
                                       bias=False,
                                       name='all_features')

            self.update_all_embeddings = tf.assign(ref=self.all_embeddings, value=all_embeddings)

        # ===============================================
        # RUN GRAPH
        # ===============================================
        var_reg.append(self.all_embeddings)

        with tf.name_scope("run"):
            ri_inputs = run_inputs

            feature_lookup = tx.Lookup(ri_inputs,
                                       seq_size=ctx_size,
                                       lookup_shape=[k_dim, embed_dim],
                                       weight_init=embed_init,
                                       name="lookup")

            self.embedding_lookup = feature_lookup
            var_reg.append(feature_lookup.weights)
            feature_lookup = feature_lookup.as_concat()
            # ===========================================================

            last_layer = feature_lookup
            h_layers = []
            for i in range(num_h):
                h_i = tx.Linear(last_layer, h_dim, h_init, bias=True, name="h_{i}_linear".format(i=i))
                h_a = tx.Activation(h_i, h_activation)
                h = tx.Compose(h_i, h_a, name="h_{i}".format(i=i))
                h_layers.append(h)
                last_layer = h
                var_reg.append(h_i.weights)

            self.h_layers = h_layers

            # feature prediction for Energy-Based Model

            f_prediction = tx.Linear(last_layer, embed_dim, f_init, bias=True, name="f_predict")
            var_reg.append(f_prediction.weights)

            # RI DECODING ===============================================

            # Shared Embeddings
            # dot product of f_predicted . all_embeddings with bias for each target word
            # TODO change sindex size to a param that model runner will feed before anything else
            # TODO figure out how to work with dynamic n_units
            run_logits = tx.Linear(f_prediction,
                                   n_units=len(sign_index),
                                   shared_weights=self.all_embeddings,
                                   transpose_weights=True,
                                   bias=logit_bias,
                                   name="logits")

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
                # labels
                labels = loss_inputs.tensor

                #  convert labels to random indices
                def labels_to_ri(x):
                    random_index_tensor = ri_tensor.gather(x)
                    sp_features = random_index_tensor.to_sparse_tensor()
                    return sp_features

                model_prediction = f_prediction.tensor

                train_loss = tx.sparse_cnce_loss(label_features=labels,
                                                 model_prediction=model_prediction,
                                                 weights=feature_lookup.weights,
                                                 noise_ratio=noise_level,
                                                 num_samples=nce_samples,
                                                 labels_to_sparse_features=labels_to_ri)


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

    def update_state(self):
        return self.update_all_embeddings
