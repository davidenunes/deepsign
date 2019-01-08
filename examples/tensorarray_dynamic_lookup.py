import tensorflow as tf
import tensorx as tx
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N = 100
M = 20
B = 4
H = 3

"""
things we can encounter in a dynamic rnn

max_sequence_length = tf.reduce_max(sequence_length)
# max number of timesteps for the current batch, this is particularly useful 
# when the sequences are being padded, in this case we can supply the seq len of each element in the batch

# PADDED BATCH          |  SEQ_LEN
# ---------------------------------------------
# [[1, 2, 3, 4, ~, ~ ]  |   [4,
#  [1, 2, 3, ~, ~, ~ ]  |    3,
#  [1, 2, 3, 4, 5, 6 ]  |    6,
#  [1, 2, 3, 4, 5, ~ ]] |    5]
# ---------------------------------------------
# in this case we are not using it, not sure if we need to
# this is not even the case, apparently 
# the tensor input shape be something like 
[batch_size, None, N]

but if I turn this into a time major tensor
it would be 

[None, batch_size, N] does this means each batch has the same number of seq len? yes it does...
"""

inputs = tx.Input(n_units=None, dtype=tf.int32)
lookup = tx.Lookup(inputs, seq_size=None, lookup_shape=[N, M])
input_seq = lookup.as_seq()

# this is a time major sequence so we can look at the number of elements
seq_size = tf.shape(input_seq)[0]

ta_input = tf.TensorArray(dtype=input_seq.dtype, size=seq_size, tensor_array_name="input_tensors")
ta_input = ta_input.unstack(input_seq)

ta_output = tf.TensorArray(dtype=tf.float32, size=seq_size, tensor_array_name="output_tensors")
init_vars = (0, ta_output)
cond = lambda i, _: tf.less(i, seq_size)


def body1(i, y):
    xt = ta_input.read(i)
    y = y.write(i, 2 * xt)
    return i + 1, y


def body2(i, y):
    xt = input_seq[i]
    y = y.write(i, 2 * xt.tensor)
    return i + 1, y


_, out1 = tf.while_loop(cond=cond, body=body1, loop_vars=init_vars)
out1 = out1.stack()

ta_output = tf.TensorArray(dtype=tf.float32, size=seq_size, tensor_array_name="output_tensors")
init_vars = (0, ta_output)
_, out2 = tf.while_loop(cond=cond, body=body2, loop_vars=init_vars)
out2 = out2.stack()

""" ********************************************************************************************
"""

ta_output = tf.TensorArray(dtype=tf.float32, size=seq_size, tensor_array_name="output_tensors")

# I cant accumulate objects inside while loop so I cant use the following in graph mode
# cells = []
# cells.append(tx.RNNCell(x0, n_units=H, previous_cell=None))
# use cell[0]
# also the states are wrong so I must use a TensorArray to pass the states

x0 = ta_input.read(0)
x0 = tx.TensorLayer(x0)
cell = tx.RNNCell(x0, M)
ta_output = ta_output.write(0, cell.tensor)

init_vars = (1, ta_output, cell.state)
cond_rnn = lambda i, *_: tf.less(i, seq_size)

print("creating rnn body")


def rnn_unroll(i, y, state):
    xt = ta_input.read(i)
    xt = tx.TensorLayer(xt)
    c = cell.reuse_with(xt, previous_state=state)
    y = y.write(i, c.tensor)
    return i + 1, y, c.state


print("done")
_, out3, last_state_dynamic = tf.while_loop(cond=cond_rnn, body=rnn_unroll, loop_vars=init_vars, name="rnn_unroll")
out3 = out3.stack()
last_h_dynamic = last_state_dynamic[0]
# last_memory_dynamic = last_state_dynamic[1]

print("creatint static graph")
last_state = cell.state
cells = [cell]
for k in range(1, 10):
    it = input_seq[k]
    cell_i = cell.reuse_with(it, previous_state=last_state)
    last_state = cell_i.state
    cells.append(cell_i)

last_h_static = cells[-1].state[0].tensor

""" ********************************************************************************************
"""

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data_writer = tf.summary.FileWriter("/tmp/tfexamples", sess.graph)

    seq_data1 = np.random.random_integers(0, N, [B, 5])
    seq_data2 = np.random.random_integers(0, N, [B, 10])

    loop_ta = sess.run(out1, {inputs.placeholder: seq_data1}, options=run_options, run_metadata=run_metadata)
    data_writer.add_run_metadata(run_metadata, tag="Loop with input TensorArray 1")
    loop_ta2 = sess.run(out1, {inputs.placeholder: seq_data2}, options=run_options, run_metadata=run_metadata)
    data_writer.add_run_metadata(run_metadata, tag="Loop with input TensorArray 2")

    loop_slice = sess.run(out2, {inputs.placeholder: seq_data1}, options=run_options, run_metadata=run_metadata)
    data_writer.add_run_metadata(run_metadata, tag="Loop with input slice 1")
    loop_slice2 = sess.run(out2, {inputs.placeholder: seq_data2}, options=run_options, run_metadata=run_metadata)
    data_writer.add_run_metadata(run_metadata, tag="Loop with input slice 2")

    # ******
    print("running")
    rnn_res, h_dynamic, h_static = sess.run([out3,
                                             last_h_dynamic,
                                             last_h_static],
                                             {inputs.placeholder: seq_data2}, options = run_options,
                                                                                        run_metadata = run_metadata)
    data_writer.add_run_metadata(run_metadata, tag="RNN UNROLL")

    np.testing.assert_array_equal(h_dynamic, h_static)
    #np.testing.assert_array_equal(c_dynamic, c_static)
