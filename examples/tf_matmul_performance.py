import tensorflow as tf
import tensorx as tx
import numpy as np
import time
import os
from tensorflow.contrib.compiler import xla

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = 10000
var_size = 500
batch_size = 20
seq_size = 30

inputs = tf.constant(np.random.randint(0, 10, size=[batch_size, seq_size]), name="inputs")
targets = tf.constant(np.random.randint(0, 10, size=[batch_size * seq_size]), name="targets")
targets = tf.one_hot(targets, input_size)

inputs = tx.TensorLayer(inputs)

with jit_scope():
    with tf.name_scope("scope1"):
        lookup = tx.Lookup(inputs, seq_size=seq_size, lookup_shape=[input_size, var_size], name="lookup")
        seq = lookup.permute_batch_time()
        seq = tx.Reshape(seq, [-1, var_size], name="flatten")
        mul1 = tx.Linear(seq, input_size, name="test_logits")
        mul2 = tx.Linear(seq,
                         n_units=input_size,
                         shared_weights=lookup.weights,
                         transpose_weights=True,
                         name="shared_embeddings")

    with tf.name_scope("scope2"):
        mul1 = mul1.reuse_with(seq)
        mul2 = mul2.reuse_with(seq)

rnd_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=mul1))
rnd_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=mul2))

config = tf.ConfigProto()
# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(os.environ["HOME"] + "/tmp/", sess.graph)
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE
    metadata = tf.RunMetadata()
    sess.run(tf.global_variables_initializer())

    t0 = time.time()
    out1, out2 = sess.run([mul1.tensor, mul2.tensor], options=options, run_metadata=metadata)
    print(time.time() - t0)
    writer.add_run_metadata(metadata, tag="{}".format(1), global_step=1)

    writer.close()

    # print(metadata.step_stats)

    # print(rnd_loss1.eval())
    # print(rnd_loss1.eval())
    # print(rnd_loss2.eval())
