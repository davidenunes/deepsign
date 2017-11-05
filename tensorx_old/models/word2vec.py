from tensorx_old.layers import FeatureInput, Embeddings, Dense
from tensorx_old.init import glorot_init,random_uniform_init
import tensorflow as tf
import math


class SkipGram:
    def __init__(self, vocab_size, embedding_dim, batch_size):
        """ Builds the graph for a Skip-gram model
        
        :param vocab_size: number of unique words
        :param embedding_dim: dimension for embedding vectors
        """
        self.n_units = vocab_size
        self.n_active = 1

        # 1-of-V encoding
        input_layer = FeatureInput(self.n_units, self.n_active, batch_size=batch_size)
        embeddings = Embeddings(input_layer, embedding_dim, act=None, bias=False)
        # output also has a 1-of-V encoding
        #output = Dense(embeddings, n_units=vocab_size, bias=True, name="out")
        self.nce_weights = tf.Variable(
            tf.truncated_normal(
                [self.n_units, embedding_dim],
                stddev=1.0 / math.sqrt(embedding_dim))
            )
        self.nce_biases = tf.Variable(tf.zeros([self.n_units]))


        self.embeddings = embeddings

        self.labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])
        self.input = input_layer
        #self.output = output


    def nce_loss(self):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=self.labels,
                           inputs=self.embeddings(),
                           num_sampled=2,
                           num_classes=self.n_units))
        return loss
