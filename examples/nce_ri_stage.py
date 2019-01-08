from tensorflow.python.ops import array_ops, math_ops, variables, candidate_sampling_ops, sparse_ops
from tensorflow.python.framework import dtypes, ops, sparse_tensor
from tensorflow.python.ops.nn import embedding_lookup_sparse, embedding_lookup, sigmoid_cross_entropy_with_logits

import tensorx.layers
import tensorx.train
from deepsign.rp.tf_utils import RandomIndexTensor
from deepsign.rp.ri import Generator, RandomIndex
from deepsign.data.transform import ris_to_sp_tensor_value

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorx as tx
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
sns.set_style("white")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.InteractiveSession()

vocab_size = 2000
k = 1000
s = 4
embed_size = 128
nce_samples = 1000

# *************************************
#   RI VOCAB GENERATION
# *************************************
generator = Generator(k, s)
ris = [generator.generate() for _ in range(vocab_size)]

ri_tensor = ris_to_sp_tensor_value(ris, k)
ri_tensor = tf.convert_to_tensor_or_sparse_tensor(ri_tensor)

# *************************************
#   DUMMY INPUT DATA
# *************************************
# batch of word sequence indices
ctx_size = 3
input_data = np.array([[0, 1, 2],
                       [0, 2, 2],
                       [1, 3, 5],
                       [3, 0, 2]])

input_labels = tf.constant(np.array([[3], [1], [10], [25]], dtype=np.int64))
input_labels = tx.TensorLayer(input_labels, n_units=1)

input_layer = tx.TensorLayer(input_data, n_units=3, dtype=tf.int64)

ri_layer = tx.TensorLayer(ri_tensor, k)
ri_inputs = tx.gather_sparse(ri_layer.tensor, input_layer.tensor)
ri_inputs = tx.TensorLayer(ri_inputs, k)
lookup = tx.Lookup(ri_inputs, ctx_size, [k, embed_size],
                   weight_init=tx.random_normal(0, 0.1), name="lookup")
feature_predict = tx.Linear(lookup, embed_size, bias=True)

all_embeddings = tx.Linear(ri_layer,
                           embed_size,
                           shared_weights=lookup.weights,
                           name="all_features",
                           bias=False)

# dot product of f_predicted . all_embeddings with bias for each target word
run_logits = tx.Linear(feature_predict, vocab_size, shared_weights=all_embeddings.tensor,
                       transpose_weights=True,
                       bias=False, name="logits")

embed_prob = tx.Activation(run_logits, tx.softmax, name="run_output")

one_hot = tx.dense_one_hot(column_indices=input_labels.tensor, num_cols=vocab_size)
val_loss = tx.categorical_cross_entropy(one_hot, run_logits.tensor)
val_loss = tf.reduce_mean(val_loss)

# *************************************
#   Testing adaptive noise
# *************************************
noise_logits = tx.Linear(lookup, k, bias=True)
adaptive_noise = tx.sample_sigmoid_from_logits(noise_logits.tensor, n=1)
adaptive_noise = tx.TensorLayer(adaptive_noise, n_units=k)
# adaptive_noise = tx.to_sparse(adaptive_noise)

# *************************************
#   INIT
# *************************************
tf.global_variables_initializer().run()
# *************************************
#   NCE STAGING
# *************************************
noise_sampler = candidate_sampling_ops.uniform_candidate_sampler(
    true_classes=input_labels.tensor,
    num_true=1,
    num_sampled=nce_samples,
    unique=True,
    range_max=vocab_size,
    seed=None)

sampled, true_expected_count, sampled_expected_count = (
    array_ops.stop_gradient(s) for s in noise_sampler)

print("adaptive sample: ", tf.shape(noise_logits.tensor).eval())

print("[noise sample shape] {}".format(tf.shape(sampled).eval()))

labels_flat = array_ops.reshape(input_labels.tensor, [-1])

true_ris = tx.gather_sparse(sp_tensor=ri_tensor, ids=labels_flat)
noise_ris = tx.gather_sparse(sp_tensor=ri_tensor, ids=sampled)

print("----")
print("[true_ri shape] {}".format(tf.shape(true_ris).eval()))
print("[noise_ri shape] {}".format(tf.shape(noise_ris).eval()))
print("----")

true_w = embedding_lookup_sparse(params=lookup.weights,
                                 sp_ids=tx.sparse_indices(true_ris),
                                 sp_weights=true_ris,
                                 combiner="sum",
                                 partition_strategy="mod")

noise_w = embedding_lookup_sparse(params=lookup.weights,
                                  sp_ids=tx.sparse_indices(noise_ris),
                                  sp_weights=noise_ris,
                                  combiner="sum",
                                  partition_strategy="mod")

print("[true_w shape] {}".format(tf.shape(true_w).eval()))
print("[noise_w shape] {}".format(tf.shape(noise_w).eval()))
print("----")
# *************************************
#   LOGITS
# *************************************
true_logits = math_ops.matmul(feature_predict.tensor, true_w, transpose_b=True)
noise_logits = math_ops.matmul(feature_predict.tensor, noise_w, transpose_b=True)
print("[true_logit shape] {}".format(tf.shape(true_logits).eval()))
print("[noise_logits shape] {}".format(tf.shape(noise_logits).eval()))

logits = array_ops.concat([true_logits, noise_logits], 1)
print("[logit shape] {}".format(tf.shape(logits).eval()))

out_labels = array_ops.concat([
    array_ops.ones_like(true_logits),
    array_ops.zeros_like(noise_logits)
], 1)

print("[labels shape] {}".format(tf.shape(out_labels).eval()))

sampled_losses = tx.binary_cross_entropy(labels=out_labels, logits=logits, name="sampled_losses")
sampled_loss = sampled_losses
sampled_loss = math_ops.reduce_sum(sampled_losses, axis=-1)

sampled_loss = tf.reduce_mean(sampled_loss + tf.nn.l2_loss(lookup.weights) * 1e-6)

tqdm.write("loss: {}".format(sampled_loss.eval()))

lr = tensorx.layers.Param(0.0005)
# opt = tf.train.RMSPropOptimizer(learning_rate=lr.tensor)
opt = tx.AMSGrad(learning_rate=lr.tensor)
# opt = tf.train.GradientDescentOptimizer(learning_rate=lr.tensor)
# = opt.minimize(sampled_loss)

# sess.run(tf.variables_initializer(opt.variables()))

model = tx.Model(
    run_inputs=input_layer,
    train_in_loss=input_labels,
    train_out_loss=sampled_loss,
    eval_out_score=val_loss
)
runner = tx.ModelRunner(model)
runner.set_session()
runner.config_optimizer(opt,
                        optimizer_params=lr,
                        gradient_op=lambda grad: tf.clip_by_norm(grad, 1.0))

avg_nce = []
avg_ppl = []
avg_loss = []

current_nce = []
current_loss = []

lr.value = 0.1
n = 1000
for i in tqdm(range(n)):

    nce_loss = runner.train(output_loss=True)
    loss = runner.eval()

    current_loss.append(loss)
    current_nce.append(nce_loss)

    if i % 10 == 0:
        avg_nce.append(np.mean(current_nce))
        avg_loss.append(np.mean(np.exp(current_loss)))
        avg_ppl.append(np.mean(current_loss))

        current_nce = []
        current_loss = []

        # if n < 50:
        #    lr.value += 0.1
        # else:
        #    lr.value -= 0.01

# plt.plot(lr_vals, y)
fig = plt.figure(1)  # an empty figure with no axes
plt.subplot(311)
plt.plot(avg_nce)
plt.title("nce loss")

# axs[0].title("nce loss")

plt.subplot(312)
plt.plot(avg_loss)  # np.exp(y_eval)#)
plt.title("cross ent loss")
sns.despine()

plt.subplot(313)
plt.plot(avg_ppl)  # np.exp(y_eval)#)
plt.title("PPL")
sns.despine()

plt.show()

print(avg_ppl)
# ax2.title("softmax x-ent loss")


# print(noise_logits.eval())
