import tensorx as tx
import tensorflow as tf


class NNLM(tx.Model):
    def __init__(self,
                 ctx_size,
                 vocab_size,
                 embed_dim,
                 batch_size,
                 h_dim,
                 embed_init=tx.random_normal(0, 0.01),
                 logit_init=tx.random_normal(0, 0.01),
                 num_h=1,
                 h_activation=tx.relu,
                 h_init=tx.relu_init,
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
        self.vocab_size = vocab_size
        self.n_gram_size = ctx_size
        self.use_dropout = use_dropout
        self.keep_prob = keep_prob
        self.logit_init = logit_init

        if self.run_inputs is None:
            self.run_inputs = tx.Input(ctx_size, dtype=tf.int32)

        if self.loss_inputs is None:
            loss_inputs = tx.Input(n_units=vocab_size, name="one_hot_labels_loss")
            self.loss_inputs = loss_inputs

        if self.eval_inputs is None:
            eval_inputs = tx.Input(n_units=vocab_size, name="one_hot_labels_eval")
            self.eval_inputs = eval_inputs

        if run_inputs.dtype != tf.int32 and run_inputs.dtype != tf.int64:
            raise TypeError(
                "Invalid dtype for input layer: expected int32 or int64 got {a} instead".format(a=run_inputs.dtype))

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
                                 batch_size,
                                 weight_init=tx.random_normal(0, 0.05))

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
        train_loss = tx.categorical_cross_entropy(self.loss_inputs.tensor, train_logits.tensor)

        # ===============================================
        # TRAIN GRAPH
        # ===============================================
        eval_loss = tx.categorical_cross_entropy(self.eval_inputs.tensor, run_logits.tensor)
        eval_tensors = tf.reduce_mean(eval_loss)

        # BUILD MODEL
        super().__init__(run_in_layers=self.run_inputs, run_out_layers=run_output,
                         train_in_layers=self.run_inputs, train_out_layers=train_output,
                         eval_in_layers=self.run_inputs, eval_out_layers=run_output,
                         train_loss_tensors=train_loss, train_loss_in=self.loss_inputs,
                         eval_tensors=eval_tensors, eval_tensors_in=self.eval_inputs)
