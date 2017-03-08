from tensorx.layers import Input, Dense, Act
from tensorx.init import glorot
import tensorflow as tf
from functools import partial

random_uniform_init = partial(tf.random_uniform, minval=-1,maxval=1)


class ANN:

    def input(self):
        """
        Returns a tensor with the input placeholder
        :return:
        """
        return self.x()

    def save(self, session, filename, step=None):
        saver = tf.train.Saver()
        # saves in name.meta
        saver.save(session, filename, global_step=step)

    def load(self, session, filename):
        saver = tf.train.Saver()
        saver.restore(session, filename)



class NRP(ANN):
    def __init__(self, k_dim, h_dim=300,h_init=glorot,h_act=tf.identity):
        """
        Creates a neural random projections model based on maximum likelihood of context outputs
        :param k_dim: dimensionality of input and output layers
        :param h_dim: dimensionality for embeddings layer
        """
        self.k_dim = k_dim
        self.h_dim = h_dim

        # model definition
        x = Input(n_units=k_dim, name="x")
        h = Dense(x, n_units=h_dim, init=h_init, name="W_f", act=h_act)
        yp = Dense(h, n_units=k_dim, init=glorot, bias=True, name="W_p")
        yn = Dense(h, n_units=k_dim, init=glorot, bias=True, name="W_n")

        self.x = x
        self.h = h
        self.yp = yp
        self.yn = yn


    def output_sample(self):
        """
        Samples from the output distribution by flipping "coins" rp_i and rn_i for each bit on
        the output layers yp_i and the output layer yn_i respectively, sampling from each layer as follows
            (y_i = +1)  if rp_i < yp_i
            (y_i = -1) if rn_i < yn_i
        :returns:
            a tensor that samples from the output distributions
        :expects:
            an input tensor to be fed
        """
        sample_pos = tf.cast(tf.less(tf.random_uniform([1, self.k_dim]), tf.sigmoid(self.yp())), tf.float32)
        sample_neg = tf.cast(tf.less(tf.random_uniform([1, self.k_dim]), tf.sigmoid(self.yn())), tf.float32)

        return sample_pos - sample_neg

    def get_loss(self, labels_p, labels_n):
        """
        Get a tensor for computing the sigmoid cross entropy with logits loss for this model
        :param labels_p: input layer with positive output labels
        :param labels_n: input layer with negative output labels
        :return:
        """
        if not (isinstance(labels_p,Input) and isinstance(labels_n,Input)):
            raise ValueError("labels need to be tensorx.layers.Input instances")

        loss_p = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_p(), logits=self.yp()))
        loss_n = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_n(), logits=self.yn()))
        loss = loss_p + loss_n

        return loss

    def get_perplexity(self, labels_p, labels_n):
        """
        Returns the Perplexity of the model which is just the 2^{cross entropy}, this is:
            an exponential of average negative log likelihood
        :param labels_p:
        :param labels_n:
        :return: a tensor that computers the perplexity of the model
        """
        return tf.pow(2.0, self.get_loss(labels_p, labels_n))

    def embedding_regularisation(self,weight):
        w_reg = tf.nn.l2_loss(self.h.weights)
        return w_reg * weight

    def output_regularisation(self,weight):
        wp_reg = tf.nn.l2_loss(self.yp.weights)
        wn_reg = tf.nn.l2_loss(self.yn.weights)
        return wp_reg * weight + wn_reg * weight


class NRPRegression(ANN):
    def __init__(self, k_dim, h_dim=300,h_init=glorot,h_act=tf.identity):
        """
        Creates a neural random projections model based on maximum likelihood of context outputs
        :param k_dim: dimensionality of input and output layers
        :param h_dim: dimensionality for embeddings layer
        """
        self.k_dim = k_dim
        self.h_dim = h_dim

        # model definition
        x = Input(n_units=k_dim, name="x")
        h = Dense(x, n_units=h_dim, init=h_init, name="W_f", act=h_act)
        y = Dense(h, n_units=k_dim, init=glorot, bias=True, name="W_o", act=Act.tanh)

        self.x = x
        self.h = h
        self.y = y

    def get_loss(self, labels_out):
        """
        Get a tensor for computing the sigmoid cross entropy with logits loss for this model
        :param labels_p: input layer with positive output labels
        :param labels_n: input layer with negative output labels
        :return:
        """
        if not isinstance(labels_out,Input):
            raise ValueError("labels need to be tensorx.layers.Input instances")

        loss_reg = tf.reduce_mean(tf.squared_difference(labels_out(),self.y()))

        return loss_reg

    def output_sample(self):
        """
        Samples from the output distribution by flipping "coins" rp_i and rn_i for each bit on
        the output layer for positve and negative entries respectively

            (y_i = +1)  if rp_i < yp_i
            (y_i = -1) if rn_i < yn_i
        :returns:
            a tensor that samples from the output distributions
        :expects:
            an input tensor to be fed
        """
        sample = tf.cast(tf.less(tf.random_uniform([1, self.k_dim]), tf.abs(self.y())), tf.float32)
        sign_sample = sample * tf.sign(self.y())

        return sign_sample


class NRPSkipReg(ANN):
    def __init__(self, k_dim, h_dim=300,h_init=random_uniform_init,h_act=tf.identity,window_reach=2):
        """
        Creates a neural random projections model based on maximum likelihood of context outputs
        :param k_dim: dimensionality of input and output layers
        :param h_dim: dimensionality for embeddings layer

        """
        self.window_reach = window_reach
        self.k_dim = k_dim
        self.h_dim = h_dim

        # model definition
        x = Input(n_units=k_dim, name="x")
        self.x = x
        h = Dense(x, n_units=h_dim, init=h_init, name="embeddings", act=h_act)
        self.h = h

        self.y = []
        for i in range(window_reach*2):
            self.y.append(Dense(h,n_units=k_dim,init=random_uniform_init,bias=True,act=Act.tanh,name="y_"+i))

    def get_loss(self, labels_out):
        """
        Get a tensor for computing the sigmoid cross entropy with logits loss for this model
        :param labels_out a list of input layers where random indexes are fed
        :return:
        """
        if len(labels_out) != self.window_reach*2:
            raise ValueError("Number of labels has to match number of context words")

        loss_reg = 0
        for i in range(self.window_reach*2):
            loss_reg += tf.reduce_mean(tf.squared_difference(labels_out[i](),self.y[i]()))

        return loss_reg

    def output_sample(self,i):
        """
        Samples from the output distribution by flipping "coins" rp_i and rn_i for each bit on
        the output layer i for positve and negative entries respectively

            (y_i = +1)  if rp_i < yp_i
            (y_i = -1) if rn_i < yn_i
        :returns:
            a tensor that samples from the output distributions
        :expects:
            an input tensor to be fed
        """
        out_layer = self.y[i]
        sample = tf.cast(tf.less(tf.random_uniform([1, self.k_dim]), tf.abs(out_layer())), tf.float32)
        sign_sample = sample * tf.sign(out_layer())

        return sign_sample
