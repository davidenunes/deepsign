import tensorflow as tf
import os
import shutil
import numpy as np

"""
saver = tf.train.import_meta_graph('results/model.ckpt-1000.meta')
graph = tf.get_default_graph()

# Finally we can retrieve tensors, operations, collections, etc.
global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
train_op = graph.get_operation_by_name('loss/train_op')
...
"""

shape = [2]

var = tf.get_variable("test_var", shape=shape, initializer=tf.random_uniform_initializer(-1, 1))
saver = tf.train.Saver()
var_init = tf.global_variables_initializer()
vars = []

if not os.path.exists("checkpoints") or not os.path.isdir("checkpoints"):
    os.makedirs("checkpoints")
try:
    with tf.Session() as sess:
        print("session 1")
        sess.run(var_init)
        for i in range(4):
            saver.save(sess, save_path="checkpoints/model_ckpt", global_step=i, write_meta_graph=False,
                       write_state=True)
            # the write_state determines whether or not to write a checkpoint file with a list of all checkpoints
            vars.append(sess.run(var))
            # I was doing var += something, appending the results of the op but saving only the first var ...
            sess.run(tf.assign_add(var, tf.random_uniform(shape, -1, 1, dtype=tf.float32)))

        assert (len(vars) == 4)

        for op in tf.global_variables():
            print(str(op.name))

    for var in vars:
        print(var)

    tf.reset_default_graph()

    var = tf.get_variable("test_var", shape=shape)
    # needs to be created after variables are created
    saver = tf.train.Saver()

    print("restore checkpoints")

    with tf.Session() as sess:
        checkpoint = "checkpoints/model_ckpt-{i}".format(i=3)
        print("restoring ", checkpoint)
        saver.restore(sess, checkpoint)
        var_i = sess.run(var)
        np.testing.assert_array_equal(var_i, vars[i])

    print("done checking checkpoints")

    # with open("test.txt", "w") as f:
    #    f.writelines("Hello world")
except Exception as e:
    # input()
    # raise
    raise e
    shutil.rmtree("checkpoints")
