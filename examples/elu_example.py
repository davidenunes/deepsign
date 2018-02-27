import tensorflow as tf
import tensorx as tx
import numpy as np
from tqdm import tqdm

n_hidden = 20
embed_dim = 10
seq_size = 2
vocab_size = 100
feature_shape = [vocab_size, embed_dim]

loss_inputs = tx.Input(1, dtype=tf.int32)
in_layer = tx.Input(seq_size, dtype=tf.int32)

lookup = tx.Lookup(in_layer, seq_size=seq_size,
                   feature_shape=feature_shape)
# [batch x seq_size * feature_shape[1]]

h = tx.Linear(lookup, n_hidden)
ha = tx.Activation(h, tx.elu)
h = tx.Compose([h, ha])


logits = tx.Linear(h, vocab_size)
out = tx.Activation(logits, tx.softmax)

labels = tx.dense_one_hot(loss_inputs.tensor, vocab_size)
loss = tf.reduce_mean(tx.categorical_cross_entropy(labels=labels, logits=logits.tensor))

# setup optimizer
optimizer = tx.AMSGrad(learning_rate=0.01)

model = tx.Model(run_in_layers=in_layer, run_out_layers=out,
                 train_in_layers=in_layer, train_out_layers=out,
                 train_loss_in=loss_inputs, train_loss_tensors=loss,
                 eval_tensors=loss, eval_tensors_in=loss_inputs)

print(model.feedable_train())

runner = tx.ModelRunner(model)
runner.config_optimizer(optimizer)

runner.init_vars()

# need to fix the runner interface to allow for lists to be received
data = np.array([[0, 1], [1, 0]])
targets = np.array([[2], [3]])

for i in tqdm(range(100000)):
    runner.train(data=data, loss_input_data=targets)

    if i % 1000 == 0:
        loss = runner.eval(data, targets)
        tqdm.write("loss: {}".format(loss))
