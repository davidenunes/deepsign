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
    active binary features. Essentially it is like a sparse binary vector
    which only receives the ids of the active entries but where we need to 
    know the original number of dimensions (n_units)
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
    """
    Merges a list of given layers using the provided merging function. Each layer can also be weighted and its outputs 
    will be the result of multiplying the layer output by the respective weight. 
    
    This is just a container that for convenience takes the output of each given layer (which is generaly a tensor), 
    and applies a merging function. 
    
    This layer also encapsulates a bias variable along with a possible activation function to be applied to the result
    of the merge.
    """
    def __init__(self,
                 layers,
                 weights=None,
                 merge_fn=tf.add_n,
                 act=None,
                 bias=False,
                 name="merge"):
        """
        :param layers: a list of layers with the same number of units to be merged according to a given function 
        :param weights: a list of weights, if provided, the number of weights must match the number of layers
        :param merge_fn: must operate on a list of tensors (default is add_n)
        :param act: activation function post merge
        :param bias: if true adds biases to this layer
        :param name: name for layer which creates a named-scope
        """

        if len(layers) < 2:
            raise Exception("Expecting a list of layers with len >= 2")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        # every layer should have the same n_units anyway
        self.n_units = layers[0].n_units
        self.name = name
        self.act = act

        with tf.variable_scope(name):
            if weights is not None:
                for i in range(len(layers)):
                    layers[i] = tf.scalar_mul(weights[i], layers[i].output)

            y = merge_fn(layers)

            if bias:
                b = tf.get_variable("b", initializer=tf.zeros([self.n_units]))
                y = tf.nn.bias_add(y, b, name="output")

            if act is not None:
                y = act(y, name="y")

        self.output = y

    def __call__(self):
        return self.output


class Embeddings:
    """
    The embeddings layer works like a dense layer and produces a "weights" variable 
    but takes a FeatureInput layer as input instead of an Input layer
    """
    def __init__(self,
                 features,
                 n_units,
                 init=random_uniform_init,
                 act=None,
                 bias=False,
                 weights=None,
                 name="embeddings"):

        if not isinstance(features, FeatureInput):
            raise ValueError("Invalid Input Layer: feature_input must be of type FeatureInput")

        self.features = features
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
                self.weights = tf.get_variable("w", initializer=init(shape=[self.features.n_units, self.n_units]))

            w = self.weights

            lookup = tf.nn.embedding_lookup(params=w, ids=features(), name="Embeddings")
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
