import tensorflow as tf
from tensorx.init import random_uniform_init


class Act:
    sigmoid = tf.sigmoid
    tanh = tf.tanh
    relu = tf.nn.relu
    elu = tf.nn.elu


class Input:
    def __init__(self,
                 n_units,
                 dtype=tf.float32,
                 name="input"):
        self.feed_key = name + ":0"
        self.n_units = n_units
        self.dtype = dtype
        self.name = name

        self.output = tf.placeholder(dtype=self.dtype,
                                     shape=[None, self.n_units],
                                     name=self.name)

    def __call__(self):
        return self.output


class FeatureInput:
    """
    Feature input creates a placeholder to receive n_active of n_units 
    active binary features
    """
    def __init__(self,
                 n_units,
                 n_active,
                 dtype=tf.float32,
                 name="input"):
        self.feed_key = name + ":0"
        self.n_units = n_units
        self.n_active = n_active

        self.dtype = dtype
        self.name = name

        self.output = tf.placeholder(dtype=self.dtype,
                                     shape=[None, self.n_active],
                                     name=self.name)

    def __call__(self):
        return self.output


class Merge:
    def __init__(self,
                 layer1,
                 layer2,
                 act=tf.add,
                 bias=False,
                 name="merge"):

        if layer1.n_units != layer2.n_units:
            raise Exception("Can only merge layers with the same number of units")

        self.name = name
        self.act = act
        y = act(layer1,layer2)

        if bias:
            b = tf.get_variable("b", initializer=tf.zeros([]))
            y = tf.nn.bias_add(y, b, name="output")

        self.output = y

    def __call__(self):
        return self.output



class Embeddings:
    def __init__(self,
                 feature_input,
                 n_units,
                 init=random_uniform_init,
                 act=None,
                 bias=False,
                 weights=None,
                 name="dense"):

        if not isinstance(feature_input, FeatureInput):
            raise ValueError("Invalid Input Layer: feature_input must be of type FeatureInput")

        self.init = init
        self.n_units = n_units
        self.name = name
        self.weights = weights

        if weights is not None:
            (_, s) = weights.get_shape()
            if s != n_units:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with tf.variable_scope(name):
            if weights is None:
                self.weights = tf.get_variable("w", initializer=init(shape=[self.feature_input.n_units, self.n_units]))

            w = self.weights

            lookup = tf.nn.embedding_lookup(params=w, ids=feature_input(), name="Embeddings")
            y = tf.reduce_sum(lookup, axis=1)

            if bias:
                b = tf.get_variable("b", initializer=tf.zeros([n_units]))
                y = tf.nn.bias_add(y, b, name="output")

            if act is not None:
                y = act(y, name="y")

            self.output = y

    def __call__(self):
        return self.output


class Dense:
    def __init__(self,
                 input_layer,
                 n_units,
                 init=random_uniform_init,
                 act=None,
                 bias=False,
                 weights=None,
                 name="dense"):
        self.init = init
        self.n_units = n_units
        self.name = name
        self.weights = weights

        if weights is not None:
            (_, s) = weights.get_shape()
            if s != n_units:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with tf.variable_scope(name):
            if weights is None:
                self.weights = tf.get_variable("w", initializer=init(shape=[input_layer.n_units, self.n_units]))

            w = self.weights
            y = tf.matmul(input_layer(), w)

            if bias:
                b = tf.get_variable("b", initializer=tf.zeros([n_units]))
                y = tf.nn.bias_add(y, b, name="o")

            if act is not None:
                y = act(y, name="y")

            self.output = y

    def __call__(self):
        return self.output
