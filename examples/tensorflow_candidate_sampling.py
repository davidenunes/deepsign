import tensorflow as tf
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler
import numpy as np
import os
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

labels = np.array([[0], [2]])
num_samples = 2
num_classes = 10

tf.enable_eager_execution()

sampled_indices = uniform_sampler(true_classes=labels,
                                  num_true=1,
                                  num_sampled=num_samples,
                                  unique=False,
                                  range_max=num_classes,
                                  seed=None)

with tf.device("/gpu:0"):
    noise_ids, target_noise_prob, noise_prob = sampled_indices

print(noise_ids)
print(target_noise_prob / num_samples)
print(noise_prob / num_samples)

print()

v = 1000
k = 10  # [1,v-1]
batch_size = 128

k = k / batch_size

p_noise = np.linspace(0, 1, num=10, endpoint=False, dtype=np.float32)
p_model = np.linspace(0, 1, num=10, endpoint=False, dtype=np.float32)


# 1 / (1 + kexp(-x)) == 1 / (1 + exp(-(x - log(k)))


def activation(p_model, p_noise):
    return tf.sigmoid(tf.log(p_model) - tf.log(p_noise) - tf.log(k))

with tf.device("/gpu:0"):
    activations = np.array([activation(pm, pn) for pn, pm in itertools.product(p_noise, p_model)])
    activations = np.nan_to_num(activations)
    activations = np.reshape(activations, [10, 10])

print(activations[0, 0])

values = np.array(list(itertools.product(p_noise, p_model)))

fig = plt.figure()

plt.contour(p_model, p_noise, activations)

plt.show()
