import tensorx as tx
import tensorflow as tf


class NNLM(tx.Model):
    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 batch_size,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.elu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 keep_prob=0.95,
                 run_inputs=None,
                 eval_inputs=None,
                 loss_inputs=None,
                 nce=False,
                 nce_samples=100):
        # NNLM PARAMS
        self.embed_init = embed_init
        self.loss_inputs = loss_inputs
        self.eval_inputs = eval_inputs
        self.run_inputs = run_inputs
        self.h_init = h_init
        self.h_activation = h_activation
        self.h_dim = h_dim
        self.num_h = num_h
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_gram_size = ctx_size
        self.use_dropout = use_dropout
        self.keep_prob = keep_prob
        self.logit_init = logit_init
        self.nce = nce
        self.nce_samples = nce_samples

        if self.run_inputs is None:
            self.run_inputs = tx.Input(ctx_size, dtype=tf.int32)

        if self.loss_inputs is None:
            self.loss_inputs = tx.Input(n_units=1, name="one_hot_labels_loss", dtype=tf.int64)

        if self.eval_inputs is None:
            self.eval_inputs = tx.Input(n_units=1, name="one_hot_labels_eval", dtype=tf.int64)

        if self.run_inputs.dtype != tf.int32 and self.run_inputs.dtype != tf.int64:
            raise TypeError(
                "Invalid dtype for input layer: expected int32 or int64 got {a} instead".format(
                    a=self.run_inputs.dtype))

        # hidden layer
        if num_h < 1:
            raise ValueError("num hidden should be >= 1")

        # ===============================================
        # RUN GRAPH
        # ===============================================

        # lookup layer
        embeddings_shape = [vocab_size, embed_dim]
        lookup_layer = tx.Lookup(self.run_inputs,
                                 ctx_size,
                                 embeddings_shape,
                                 batch_size=None,
                                 weight_init=embed_init)

        out_layer = lookup_layer
        h_layers = []
        for i in range(num_h):
            h_i = tx.Linear(out_layer, h_dim, h_init, bias=True, name="h_{i}_linear".format(i=i))
            h_a = tx.Activation(h_i, h_activation)
            h = tx.Compose([h_i, h_a], name="h_{i}".format(i=i))
            h_layers.append(h_i)
            out_layer = h

        run_logits = tx.Linear(out_layer, vocab_size, init=logit_init, bias=True, name="run_logits")
        run_output = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================

        out_layer = lookup_layer

        # add dropout between each layer
        for layer in h_layers:
            h = layer.reuse_with(out_layer)
            if use_dropout:
                h = tx.Dropout(h, keep_prob=keep_prob)
            out_layer = h

        # share logits but connected to train graph (which contains dropout)
        train_logits = run_logits.reuse_with(out_layer, name="train_logits")
        train_output = tx.Activation(train_logits, tx.softmax, name="train_output")

        if self.nce:
            # learns unigram dist during training
            # sampled_values = tf.nn.learned_unigram_candidate_sampler(self.loss_inputs.tensor,
            #                                                         num_true=1,
            #                                                         num_sampled=self.nce_samples,
            #                                                         unique=True,
            #                                                         range_max=self.vocab_size)
            # labels = tf.reshape(self.loss_inputs.tensor, [-1])

            sampled_values = tf.nn.uniform_candidate_sampler(self.loss_inputs.tensor, 1, self.nce_samples,
                                                             unique=True, range_max=self.vocab_size)

            # one_hot = tx.dense_one_hot(column_indices=self.loss_inputs.tensor, num_cols=self.vocab_size)
            train_loss = tf.nn.nce_loss(weights=tf.transpose(train_logits.weights),
                                        biases=train_logits.bias,
                                        inputs=out_layer.tensor,
                                        labels=self.loss_inputs.tensor,
                                        num_sampled=self.nce_samples,
                                        num_classes=self.vocab_size,
                                        num_true=1,
                                        sampled_values=sampled_values)
        else:
            one_hot = tx.dense_one_hot(column_indices=self.loss_inputs.tensor, num_cols=self.vocab_size)
            train_loss = tx.categorical_cross_entropy(one_hot, train_logits.tensor)

        train_loss = tf.reduce_mean(train_loss)

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        one_hot = tx.dense_one_hot(column_indices=self.eval_inputs.tensor, num_cols=self.vocab_size)
        eval_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)

        eval_tensors = tf.reduce_mean(eval_loss)

        # BUILD MODEL
        super().__init__(run_in_layers=self.run_inputs, run_out_layers=run_output,
                         train_in_layers=self.run_inputs, train_out_layers=train_output,
                         eval_in_layers=self.run_inputs, eval_out_layers=run_output,
                         train_loss_tensors=train_loss, train_loss_in=self.loss_inputs,
                         eval_tensors=eval_tensors, eval_tensors_in=self.eval_inputs)


class NRP(tx.Model):
    def __init__(self,
                 ctx_size,
                 ri_dim,
                 embed_dim,
                 batch_size,
                 h_dim,
                 embed_init=tx.random_normal(0, 0.01),
                 logit_init=tx.random_normal(0, 0.01),
                 num_h=1,
                 h_activation=tx.relu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 keep_prob=0.1,
                 run_inputs=None,
                 eval_inputs=None,
                 loss_inputs=None):
        # NNLM PARAMS
        self.embed_init = embed_init
        self.loss_inputs = loss_inputs
        self.eval_inputs = eval_inputs
        self.run_inputs = run_inputs
        self.h_init = h_init
        self.h_activation = h_activation
        self.h_dim = h_dim
        self.num_h = num_h
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.ri_dim = ri_dim
        self.ctx_size = ctx_size
        self.use_dropout = use_dropout
        self.keep_prob = keep_prob
        self.logit_init = logit_init

        if self.run_inputs is None:
            self.run_inputs = tx.SparseInput(self.ri_dim, dtype=tf.float32)
        elif not isinstance(self.run_inputs, tx.SparseInput):
            raise TypeError(
                "expected run_inputs to be a SparseInput layer got {} instead".format(type(self.run_inputs)))

        if self.loss_inputs is None:
            loss_inputs = tx.Input(self.ri_dim * 2, dtype=tf.float32, name="ri_labels_loss")
            self.loss_inputs = loss_inputs

        if self.eval_inputs is None:
            eval_inputs = tx.Input(n_units=self.ri_dim * 2, name="ri_labels_eval")
            self.eval_inputs = eval_inputs

        # hidden layer
        if num_h < 1:
            raise ValueError("num hidden should be >= 1")

        # ===============================================
        # RUN GRAPH
        # ===============================================

        # lookup layer
        embeddings_shape = [ri_dim, embed_dim]
        lookup_layer = tx.Lookup(self.run_inputs,
                                 ctx_size,
                                 embeddings_shape,
                                 batch_size,
                                 weight_init=embed_init)

        self.lookup = lookup_layer

        out_layer = lookup_layer
        h_layers = []
        for i in range(num_h):
            h_i = tx.Linear(out_layer, h_dim, h_init, bias=True, name="h_{i}_linear".format(i=i))
            h_a = tx.Activation(h_i, h_activation)
            h = tx.Compose([h_i, h_a], name="h_{i}".format(i=i))
            h_layers.append(h_i)
            out_layer = h

        # NRP uses random projections so we compare them with the actual random projections and use sigmoids
        # to model each feature probability independently
        # 2 classes of probabilities for positive entries and negative entries (ri_size*2)
        run_logits = tx.Linear(out_layer, ri_dim * 2, init=logit_init, bias=True, name="run_logits")
        run_output = tx.Activation(run_logits, tx.sigmoid, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================

        out_layer = lookup_layer

        # add dropout between each layer
        for layer in h_layers:
            h = layer.reuse_with(out_layer)
            if use_dropout:
                h = tx.Dropout(h, keep_prob=keep_prob)
            out_layer = h

        # share logits but connected to train graph (which contains dropout)
        train_logits = run_logits.reuse_with(out_layer, name="train_logits")
        train_output = tx.Activation(train_logits, tx.sigmoid, name="train_output")
        # we need to use binary cross entropy instead of categorical because we consider probabilities independent
        train_loss = tx.binary_cross_entropy(self.loss_inputs.tensor, train_logits.tensor)

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        eval_loss = tx.binary_cross_entropy(self.eval_inputs.tensor, run_logits.tensor)
        eval_tensors = tf.reduce_mean(eval_loss)

        # BUILD MODEL
        super().__init__(run_in_layers=self.run_inputs, run_out_layers=run_output,
                         train_in_layers=self.run_inputs, train_out_layers=train_output,
                         eval_in_layers=self.run_inputs, eval_out_layers=run_output,
                         train_loss_tensors=train_loss, train_loss_in=self.loss_inputs,
                         eval_tensors=eval_tensors, eval_tensors_in=self.eval_inputs)


class LSTMLM(tx.Model):
    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 batch_size,
                 h_dim,
                 embed_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 logit_init=tx.random_uniform(minval=-0.01, maxval=0.01),
                 num_h=1,
                 h_activation=tx.elu,
                 h_init=tx.he_normal_init,
                 use_dropout=False,
                 keep_prob=0.95,
                 run_inputs=None,
                 eval_inputs=None,
                 loss_inputs=None,
                 nce=False,
                 nce_samples=100):
        # NNLM PARAMS
        self.embed_init = embed_init
        self.loss_inputs = loss_inputs
        self.eval_inputs = eval_inputs
        self.run_inputs = run_inputs
        self.h_init = h_init
        self.h_activation = h_activation
        self.h_dim = h_dim
        self.num_h = num_h
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_gram_size = ctx_size
        self.use_dropout = use_dropout
        self.keep_prob = keep_prob
        self.logit_init = logit_init
        self.nce = nce
        self.nce_samples = nce_samples

        if self.run_inputs is None:
            self.run_inputs = tx.Input(ctx_size, dtype=tf.int32)

        if self.loss_inputs is None:
            self.loss_inputs = tx.Input(n_units=1, name="one_hot_labels_loss", dtype=tf.int64)

        if self.eval_inputs is None:
            self.eval_inputs = tx.Input(n_units=1, name="one_hot_labels_eval", dtype=tf.int64)

        if self.run_inputs.dtype != tf.int32 and self.run_inputs.dtype != tf.int64:
            raise TypeError(
                "Invalid dtype for input layer: expected int32 or int64 got {a} instead".format(
                    a=self.run_inputs.dtype))

        # hidden layer
        if num_h < 1:
            raise ValueError("num hidden should be >= 1")

        # ===============================================
        # RUN GRAPH
        # ===============================================

        # lookup layer
        embeddings_shape = [vocab_size, embed_dim]
        lookup_layer = tx.Lookup(self.run_inputs,
                                 ctx_size,
                                 embeddings_shape,
                                 batch_size=None,
                                 weight_init=embed_init)

        # LSTM
        lookup_to_seq = tf.reshape(lookup_layer.tensor, [-1, ctx_size, embed_dim])
        cell = tf.nn.rnn_cell.LSTMCell(num_units=h_dim, state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(cell, lookup_to_seq, dtype=tf.float32)
        val = tf.transpose(val, [1, 0, 2])
        last = val[-1]
        out_layer = tx.TensorLayer(last, h_dim)



        out_layer = lookup_layer
        h_layers = []
        for i in range(num_h):
            h_i = tx.Linear(out_layer, h_dim, h_init, bias=True, name="h_{i}_linear".format(i=i))
            h_a = tx.Activation(h_i, h_activation)
            h = tx.Compose([h_i, h_a], name="h_{i}".format(i=i))
            h_layers.append(h_i)
            out_layer = h

        run_logits = tx.Linear(out_layer, vocab_size, init=logit_init, bias=True, name="run_logits")
        run_output = tx.Activation(run_logits, tx.softmax, name="run_output")

        # ===============================================
        # TRAIN GRAPH
        # ===============================================

        out_layer = lookup_layer

        # add dropout between each layer
        for layer in h_layers:
            h = layer.reuse_with(out_layer)
            if use_dropout:
                h = tx.Dropout(h, keep_prob=keep_prob)
            out_layer = h

        # share logits but connected to train graph (which contains dropout)
        train_logits = run_logits.reuse_with(out_layer, name="train_logits")
        train_output = tx.Activation(train_logits, tx.softmax, name="train_output")

        if self.nce:
            # learns unigram dist during training
            # sampled_values = tf.nn.learned_unigram_candidate_sampler(self.loss_inputs.tensor,
            #                                                         num_true=1,
            #                                                         num_sampled=self.nce_samples,
            #                                                         unique=True,
            #                                                         range_max=self.vocab_size)
            # labels = tf.reshape(self.loss_inputs.tensor, [-1])

            sampled_values = tf.nn.uniform_candidate_sampler(self.loss_inputs.tensor, 1, self.nce_samples,
                                                             unique=True, range_max=self.vocab_size)

            # one_hot = tx.dense_one_hot(column_indices=self.loss_inputs.tensor, num_cols=self.vocab_size)
            train_loss = tf.nn.nce_loss(weights=tf.transpose(train_logits.weights),
                                        biases=train_logits.bias,
                                        inputs=out_layer.tensor,
                                        labels=self.loss_inputs.tensor,
                                        num_sampled=self.nce_samples,
                                        num_classes=self.vocab_size,
                                        num_true=1,
                                        sampled_values=sampled_values)
        else:
            one_hot = tx.dense_one_hot(column_indices=self.loss_inputs.tensor, num_cols=self.vocab_size)
            train_loss = tx.categorical_cross_entropy(one_hot, train_logits.tensor)

        train_loss = tf.reduce_mean(train_loss)

        # ===============================================
        # EVAL GRAPH
        # ===============================================
        one_hot = tx.dense_one_hot(column_indices=self.eval_inputs.tensor, num_cols=self.vocab_size)
        eval_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)

        eval_tensors = tf.reduce_mean(eval_loss)

        # BUILD MODEL
        super().__init__(run_in_layers=self.run_inputs, run_out_layers=run_output,
                         train_in_layers=self.run_inputs, train_out_layers=train_output,
                         eval_in_layers=self.run_inputs, eval_out_layers=run_output,
                         train_loss_tensors=train_loss, train_loss_in=self.loss_inputs,
                         eval_tensors=eval_tensors, eval_tensors_in=self.eval_inputs)