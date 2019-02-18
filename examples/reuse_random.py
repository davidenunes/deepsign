import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

rnd = tf.random_uniform([2])

rnd2 = tf.identity(rnd)

r1, r2 = sess.run([rnd, rnd2])
r1, r2 = sess.run([rnd, rnd])
print(r1)
print(r2)
