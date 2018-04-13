from deepsign.rp.ri import Generator, RandomIndex
from deepsign.rp.tf_utils import to_sparse_tensor_value
import tensorx as tx
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import candidate_sampling_ops as sampling_ops

tf.InteractiveSession()

vocab_size = 20
k = 10
s = 4
embed_size = 6

generator = Generator(k, s)
ris = [generator.generate() for _ in range(vocab_size)]
# ri_tensor = RandomIndexTensor.from_ri_list(ris, k, s)
ri_tensor = to_sparse_tensor_value(ris, k)
ri_tensor = tf.convert_to_tensor_or_sparse_tensor(ri_tensor)

num_samples = 4
num_true = 1
labels = np.array([[0]])

labels_flat = tf.reshape(labels, [-1])

sampled_values = sampling_ops.uniform_candidate_sampler(
    true_classes=labels,
    num_true=num_true,
    num_sampled=num_samples,
    unique=True,
    range_max=vocab_size,
    seed=None)

sampled, true_expected_count, sampled_expected_count = (
    tf.stop_gradient(s) for s in sampled_values)
sampled = tf.cast(sampled, tf.int64)

all_ids = tf.concat([labels_flat, sampled], 0)

all_ris = tx.gather_sparse(ri_tensor, all_ids)

# Retrieve the true weights and the logits of the sampled weights.

# weights shape is [num_classes, dim]
ri_layer = tx.TensorLayer(ri_tensor, k)
l = tx.Linear(ri_layer, embed_size, init=tx.random_normal(0, 1))
weights = l.weights

sp_values = all_ris
sp_indices = tx.sparse_indices(sp_values)

all_w = tf.nn.embedding_lookup_sparse(
    weights, sp_indices, sp_values, combiner="sum")

tf.global_variables_initializer().run()
print("labels flat: ", labels_flat.eval())
print("all labels: ", all_ids.eval())

print("ri_tensor \n", all_ris.eval())

print("weights \n", weights.eval())
print("sp_indices \n,", sp_indices.eval())
print("sp_values \n,", sp_values.eval())
print("retrieved w \n", all_w.eval())
