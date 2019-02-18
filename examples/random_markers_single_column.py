import numpy as np
import tensorflow as tf

import tensorx as tx
from deepsign.data.iterators import batch_it, shuffle_it, repeat_it

N = 10000
M = 1000

num_true = N // 2
C = np.random.randint(M)
R = np.random.choice(N, num_true)
# print("column: ", C)

labels = np.zeros(shape=(N, 1))
labels[R, 0] = 1
# print("labels\n", labels)

v = np.random.randint(2, size=[N, M])

v[..., C] = 0
v[R, C] = 1

# print("data:\n", v)
# data pipeline
batch_size = 1
epochs = 4

data = np.concatenate([v, labels], -1)

data = repeat_it(data, 2)

data = shuffle_it(iter(data), buffer_size=batch_size * 4)
data = batch_it(data, batch_size)

label_layer = tx.Input(1)
in_layer = tx.Input(M)

f1 = tx.FC(in_layer, 512, activation=tf.nn.tanh)
f2 = tx.FC(f1, 512, activation=tf.nn.relu)
fm = tx.Highway(f1, f2, carry_gate=True)

out = tx.Linear(f2, 1)
out_prob = tx.Activation(out, fn=tx.sigmoid)

loss = tx.binary_cross_entropy(labels=label_layer.tensor, logits=out.tensor)

model = tx.Model(run_inputs=in_layer,
                 run_outputs=out_prob,
                 train_in_loss=label_layer,
                 train_out_loss=loss)

runner = tx.ModelRunner(model)
runner.config_optimizer(optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
runner.init_vars()

for data_batch in data:
    data_batch = np.array(data_batch)
    ctx_vector = data_batch[:, :-1]
    label = data_batch[:, -1:]

    loss = runner.train(ctx_vector, label, output_loss=True)
    print(np.mean(loss))

"""
# prediction = tf.argmax(h0, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"""
