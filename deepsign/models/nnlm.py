import tensorx as tx
import tensorflow as tf


class NNLM(tx.Model):
    def __init__(self, n_gram_size, vocab_size, embed_dim, batch_size, h_dim, num_hidden=1,
                 h_activation=tx.relu,
                 h_init=tx.relu_init,
                 inputs=None, eval_inputs=None, loss_inputs=None):
        # NNLM PARAMS
        self.loss_inputs = loss_inputs
        self.eval_inputs = eval_inputs
        self.h_init = h_init
        self.h_activation = h_activation
        self.h_dim = h_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_gram_size = n_gram_size

        """ ===============================================================
        MAIN GRAPH       
        =============================================================== """
        if inputs is None:
            inputs = tx.Input(n_gram_size, dtype=tf.int32)

        # lookup layer
        embed_shape = [vocab_size, embed_dim]
        lookup = tx.Lookup(inputs, n_gram_size, embed_shape, batch_size, init=tx.random_normal(0, 0.05))

        # hidden layer
        if num_hidden < 1:
            raise ValueError("num hidden should be >= 1")

        last_layer = lookup
        for _ in range(num_hidden):
            h_linear = tx.Linear(last_layer, h_dim, h_init, bias=True)
            h_layer = tx.Activation(h_linear, h_activation)
            last_layer = h_layer

        # output
        logits = tx.Linear(last_layer, vocab_size, bias=True)
        output = tx.Activation(logits, tx.softmax)

        """ ===============================================================
        EVAL           
        =============================================================== """
        if eval_inputs is None:
            eval_inputs = tx.Input(n_units=vocab_size, name="one_hot_labels_eval")
            self.eval_inputs = eval_inputs
            loss = tx.categorical_cross_entropy(eval_inputs.tensor, logits.tensor)

        eval_tensors = tf.reduce_mean(loss)

        """ ===============================================================
        LOSS           
        =============================================================== """
        if loss_inputs is None:
            loss_inputs = tx.Input(n_units=vocab_size, name="one_hot_labels_loss")
            self.loss_inputs = loss_inputs
        loss = tx.categorical_cross_entropy(loss_inputs.tensor, logits.tensor)

        super().__init__(run_in_layers=inputs, run_out_layers=output,
                         train_loss_tensors=loss, train_loss_in=loss_inputs,
                         eval_tensors=eval_tensors, eval_tensors_in=eval_inputs)
