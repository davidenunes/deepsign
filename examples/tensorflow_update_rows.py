"""Example for updating the rows of a variable in tensorflow
one can use this in a parallel setting (training models in parallel with MPI for example)

"""

import tensorflow as tf

update_tensor = tf.constant([0, 0, 0, 0])

a = tf.Variable(initial_value=[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
b = tf.scatter_update(ref=a, indices=[0], updates=[update_tensor])
init = tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)

    print("a = ", s.run(a))
    print("updated = ", s.run(b))
    print("a = ", s.run(a))
