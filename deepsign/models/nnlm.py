import tensorx as tx
import tensorflow as tf


class NNLM(tx.Model):
    def __init__(self, n_gram_size, vocab_size, embed_dim, batch_size, h_dim,
                 h_activation=tx.relu,
                 h_init=tx.relu_init,
                 inputs=None, loss_inputs=None):
        # NNLM PARAMS
        self.loss_inputs = loss_inputs
        self.h_init = h_init
        self.h_activation = h_activation
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_gram_size = n_gram_size

        """ ***************************************************************
        NNLM MODEL DEFINITION          
        *************************************************************** """
        if inputs is None:
            inputs = tx.Input(n_gram_size, dtype=tf.int32)

        # lookup layer
        embed_shape = [vocab_size, embed_dim]
        lookup = tx.Lookup(inputs, n_gram_size, embed_shape, batch_size)

        # hidden layer
        h_linear = tx.Linear(lookup, h_dim, h_init, bias=True)
        h_layer = tx.Activation(h_linear, h_activation)

        # output
        logits = tx.Linear(h_layer, vocab_size, bias=True)
        output = tx.Activation(logits, tx.softmax)

        """ ***************************************************************
        NNLM MODEL LOSS FUNCTION          
        *************************************************************** """
        if loss_inputs is None:
            loss_inputs = tx.Input(n_units=vocab_size, name="one_hot_words")
        loss = tx.categorical_cross_entropy(loss_inputs.tensor, logits.tensor)

        super().__init__(inputs=inputs, outputs=output, loss_tensors=loss, loss_inputs=loss_inputs)
