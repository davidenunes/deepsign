import tensorx as tx
import tensorflow as tf
from deepsign.data.transform import ris_to_sp_tensor_value
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
                 run_inputs,
                 label_inputs,
                 eval_label_input,
                 ctx_size,
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
                 use_nce=False,
                 nce_samples=2,
                 nce_noise_amount=0.1,
                 noise_input=None,
                 ):

        self.embed_dim = embed_dim

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
                           activation=h_activation,
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
                last_layer = tx.Dropout(feature_lookup, probability=keep_prob)
            else:
                last_layer = feature_lookup

            # add dropout between each layer
            for layer in h_layers:
                h = layer.reuse_with(last_layer)
                if use_dropout:
                    h = tx.Dropout(h, probability=keep_prob)
                last_layer = h

            f_prediction = f_prediction.reuse_with(last_layer)

            train_logits = run_logits.reuse_with(f_prediction, name="train_logits")
            train_embed_prob = tx.Activation(train_logits, tx.softmax, name="train_output")

            #  convert labels to random indices
            model_prediction = f_prediction.tensor

            if use_nce:
                train_loss = tx.sparse_cnce_loss(label_features=label_inputs.tensor,
                                                 noise_features=noise_input.tensor,
                                                 model_prediction=model_prediction,
                                                 weights=feature_lookup.weights,
                                                 num_samples=nce_samples,
                                                 noise_ratio=nce_noise_amount
                                                 )
            else:
                one_hot_dense = tx.dense_one_hot(column_indices=label_inputs[0].tensor, num_cols=label_inputs[1].tensor)
                train_loss = tx.categorical_cross_entropy(one_hot_dense, train_logits.tensor)

                train_loss = tf.reduce_mean(train_loss)

            if l2_loss:
                losses = [tf.nn.l2_loss(var) for var in var_reg]
                train_loss = train_loss + l2_loss_coef * tf.add_n(losses)

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        with tf.name_scope("eval"):
            one_hot_dense = tx.dense_one_hot(column_indices=eval_label_input[0].tensor, num_cols=label_inputs[1].tensor)
            train_loss = tx.categorical_cross_entropy(one_hot_dense, train_logits.tensor)
            eval_loss = tx.categorical_cross_entropy(one_hot_dense, run_logits.tensor)
            eval_loss = tf.reduce_mean(eval_loss)

        if use_nce:
            train_loss_in = [label_inputs, noise_input]
        else:
            train_loss_in = label_inputs

        # BUILD MODEL
        super().__init__(run_inputs=run_inputs, run_outputs=embed_prob,
                         train_inputs=run_inputs, train_outputs=train_embed_prob,
                         eval_inputs=run_inputs, eval_outputs=embed_prob,
                         train_out_loss=train_loss, train_in_loss=train_loss_in,
                         eval_out_score=eval_loss, eval_in_score=eval_label_input,
                         update_inputs=ri_tensor_input)

    def update_state(self):
        # takes the value from the update_in_layers and updates the variable cache
        return self.all_embeddings.tensor
