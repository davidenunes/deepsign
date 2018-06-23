from tensorflow.python.ops import array_ops, math_ops, variables, candidate_sampling_ops, sparse_ops
from tensorflow.python.framework import dtypes, ops, sparse_tensor
from tensorflow.python.ops.nn import embedding_lookup_sparse, embedding_lookup, sigmoid_cross_entropy_with_logits
import tensorx as tx

from tensorx.layers import Layer
from tensorx import *


class LookupNCE(Layer):
    """ Lookup Noise Contrastive Estimation Layer

    This layer implements tensorflow nce loss but instead of
    using dedicated weights and bias, it can receive those shared
    variables like other layers. It outputs a single tensor that can
    be used as a NCE loss function

    Also another difference is that if it receives an input and a lookup
    index layer as inputs. The input is the output of a neural network
    meant to be used to compute the logits.

    The lookup tensor is the same as the input for a lookup layer, this is
    either a dense tensor with indices or a sparse tensor with a batch of
    of lookups for the true class to be used in NCE
    """
    def __init__(self,
                 layer,
                 n_units,
                 true_lookups,
                 bias=False,
                 weight_init=random_uniform(),
                 shared_weights=None,
                 shared_bias=None):
        self.weigh_init = weight_init
        self.bias =bias
        self.n_units =n_units
        self.true_lookups = true_lookups

        self.shared_weights= shared_weights
        self.shared_bias = shared_bias





