import tensorx as tx
import tensorflow as tf
import functools


class NNLM:
    """
    Args:
        ngram_size: size of input ngram_size-gram (if 4-grams are being used ngram_size=3 usually)
        vocab_size: vocab size
        embed_size: embedding dimension size
        batch_size: how many ngram_size-grams per batch
        h_dim: number of hidden units
        dropout_prob: probability for dropout
    """

    def __init__(self, ngram_size, vocab_size, embed_dim, batch_size, h_dim):
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self.embed_size = embed_dim
        self.embed_shape = [vocab_size, embed_dim]
        self.batch_size = batch_size
        self.h_dim = h_dim
        self.h_init = tx.relu_init

        self.inputs = tx.Input(self.ngram_size, dtype=tf.int32)
        self.lookup = tx.Lookup(self.inputs, self.ngram_size, self.embed_shape, self.batch_size)

        self.w_h = tx.Linear(self.lookup, self.h_dim, self.h_init, bias=True)
        self.h = tx.Activation(self.w_h, tx.relu)

        self.logits = tx.Linear(self.h, vocab_size, bias=True)
        self.output = tx.Activation(self.logits,tx.softmax)


    def loss(self,one_hot_labels):
        """ Creates the loss function for this model
        :param one_hot_labels: [batch_size, vocab_size] target one-hot-encoded labels.
        """
        return tx.categorical_cross_entropy(one_hot_labels,self.logits.tensor)
