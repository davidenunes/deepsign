import tensorflow as tf
import tensorx as tx
import numpy as np
from tensorflow.contrib.distributions import kl_divergence

tf.InteractiveSession()

# dummy values
model_1 = tf.constant([[1., 0., 0., 0.]])

model_2 = tf.constant([[0., 0., 1., 1.]])
logits_2 = tx.logit(model_2)

labels1 = tf.constant([[1., 0., 0., 0.]])
labels2 = tf.constant([[0., 0., 1., 1.]])

h1 = kl_divergence(model_1, labels1)
h2 = kl_divergence(model_2, labels2)

print(h1.eval())
print(h2.eval())
