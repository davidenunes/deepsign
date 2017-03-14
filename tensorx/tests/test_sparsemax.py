from tensorx.sparsemax import sparsemax, sparsemax_loss
import tensorflow as tf
import numpy as np


input1 = tf.placeholder(shape=[None,6],name="input",dtype=tf.float32)
input2 = tf.placeholder(shape=[None,6],name="input",dtype=tf.float32)

w = tf.Variable(tf.random_uniform([6,4],minval=-1,maxval=1))
init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

r = session.run(w)
#print(r)
sm = session.run(sparsemax(w))
#print(sm)


v_1 = np.zeros([6])
v_1[0] = 2
v_1[2] = 0
v_1[3] = 0
v_1[4] = 2

v_2 = np.zeros([6])
v_2[0] = 0.5
v_2[2] = 0
v_2[3] = 0
v_2[4] = 0.5


print("output: ", session.run(sparsemax(input1),feed_dict={input1:[v_1]}))

print("expected: ",v_2)
print("error: ",session.run(sparsemax_loss(input1,sparsemax(input1),input2),feed_dict={input1:[v_1],
                                                                             input2:[v_2]}))
# this does not behave as I expected I need multiple independent labels not normlised outputs



session.close()