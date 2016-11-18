"""
Implementation for Holographic Network
using random projections and sparse units
"""

import numpy as np
import tensorflow as tf
from tensorx.parts.core import NeuralNetwork, Layer


# vector of w is weights from input->hidden
# vectors normalised to unit length (euclidean norm) after training


# weight init in w2v http://building-babylon.net/2015/07/13/word2vec-weight-initialisation/
# hidden -> output = 0
# input -> hidden xavier init? [-1/2n, 1/2n] n is rank of hidden

# w2v cbow all words are projected into the same position (vectors are averaged)

class HoloNet(NeuralNetwork):
    def __init__(self, x, x_dim, h_dim=300):
        if not isinstance(x, tf.Tensor):
            raise ValueError("x must be a tensor")

        input_layer = Layer(n_units=x_dim, activation=x)
        self.nn = NeuralNetwork(input_layer)

        hidden_layer = Layer(n_units=h_dim, activation=tf.nn.re)
        output_layer = Layer(n_units=x_dim, activation=tf.nn.tanh)





