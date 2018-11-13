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
                 vocab_size,
                 k_dim,
                 ri_tensor_input,
                 embed_dim,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.relu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 embed_dropout=False,
                 keep_prob=0.95,
                 l2_loss=False,
                 l2_loss_coef=1e-5,
                 f_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 nce_samples=2,
                 nce_noise_amount=0.1
                 ):

        self.embed_dim = embed_dim

        run_inputs = tx.SparseInput(k_dim, name="sparse_context")
        loss_sparse_labels = tx.SparseInput(k_dim, name="sparse_labels")
        loss_sparse_noise = tx.SparseInput(k_dim, name="sparse_noise")
        eval_inputs = tx.Input(n_units=1)

        if isinstance(ri_tensor_input, (tx.Input, tx.SparseInput)):
            if ri_tensor_input.n_units != k_dim:
                raise ValueError("ri_tensor_input.n_units != k_dim: {} != {}".format(ri_tensor_input.n_units, k_dim))
        else:
            raise TypeError("invalid ri_tensor_input: must be an Input or SparseInput, got {} instead ".format(
                type(ri_tensor_input)))

        var_reg = []

        # ===============================================
        # RUN GRAPH
        # ===============================================

        with tf.name_scope("run"):

            feature_lookup = tx.Lookup(run_inputs,
                                       seq_size=ctx_size,
                                       lookup_shape=[k_dim, embed_dim],
                                       weight_init=embed_init,
                                       name="lookup")

            self.embeddings = feature_lookup
            var_reg.append(feature_lookup.weights)
            feature_lookup = feature_lookup.as_concat()
            # ===========================================================
            with tf.name_scope("cache_embeddings"):
                # ris = [sign_index.get_ri(sign_index.get_sign(i)) for i in range(len(sign_index))]
                # self.all_ris = ris_to_sp_tensor_value(ri_seq=ris,
                #                                      dim=sign_index.generator.dim,
                #                                      all_positive=not sign_index.generator.symmetric)

                all_embeddings = tx.Linear(ri_tensor_input,
                                           n_units=self.embed_dim,
                                           shared_weights=self.embeddings.weights,
                                           bias=False,
                                           name='all_features')

                # caches all embedding computation for run/eval
                self.all_embeddings = tx.VariableLayer(all_embeddings, trainable=False)
            # ===========================================================
            last_layer = feature_lookup
            h_layers = []
            for i in range(num_h):
                hi = tx.FC(last_layer,
                           n_units=h_dim,
                           fn=h_activation,
                           weight_init=h_init,
                           name="h_{i}".format(i=i))
                h_layers.append(hi)
                last_layer = hi
                var_reg.append(hi.linear.weights)

            self.h_layers = h_layers

            # feature prediction for Energy-Based Model

            f_prediction = tx.Linear(last_layer, embed_dim, f_init, bias=True, name="f_predict")
            var_reg.append(f_prediction.weights)

            # RI DECODING ===============================================
            # shape is (?,?) because batch size is unknown and vocab size is unknown
            # when we build the graph
            run_logits = tx.Linear(f_prediction,
                                   n_units=None,
                                   shared_weights=self.all_embeddings.variable,
                                   transpose_weights=True,
                                   bias=False,
                                   name="logits")

            # ===========================================================
            embed_prob = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================
        with tf.name_scope("train"):
            if use_dropout and embed_dropout:
                feature_lookup = feature_lookup.reuse_with(run_inputs)
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

            #  convert labels to random indices
            model_prediction = f_prediction.tensor

            train_loss = tx.sparse_cnce_loss(label_features=loss_sparse_labels.tensor,
                                             noise_features=loss_sparse_noise.tensor,
                                             model_prediction=model_prediction,
                                             weights=feature_lookup.weights,
                                             num_samples=nce_samples,
                                             noise_ratio=nce_noise_amount
                                             )

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
                         train_loss_tensors=train_loss, train_loss_in=[loss_sparse_labels, loss_sparse_noise],
                         eval_tensors=eval_loss, eval_tensors_in=eval_inputs,
                         update_in_layers=ri_tensor_input)

    def update_state(self):
        # takes the value from the update_in_layers and updates the variable cache
        return self.all_embeddings.tensor
