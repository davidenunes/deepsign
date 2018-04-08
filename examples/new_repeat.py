import tensorflow as tf
import tensorx as tx
import numpy as np
import time


def repeat_loop(elems, counts, name=None):
    """Repeat integers given by range(len(counts)) each the given number of times.
    Basically uses TensorArray + Concat like I was doing earlier, this is slow as hell though.

    Example behavior:
    [0, 1, 2, 3] -> [1, 2, 2, 3, 3, 3]
    Args:
      counts: 1D tensor with dtype=int32.
      name: optional name for operation.
    Returns:
      1D tensor with dtype=int32 and dynamic length giving the repeated integers.
    """
    with tf.name_scope(name, 'repeat_range', [counts]) as scope:
        counts = tf.convert_to_tensor(counts, name='counts')
        elems = tf.convert_to_tensor(elems, name='elems')

        dtype = elems.dtype

        def cond(unused_output, i):
            return i < size

        def body(output, i):
            value = tf.tile(tf.expand_dims(elems[i], -1), tf.expand_dims(counts[i], -1))
            return (output.write(i, value), i + 1)

        size = tf.shape(counts)[0]
        init_output_array = tf.TensorArray(
            dtype=dtype, size=size, infer_shape=False)
        output_array, num_writes = tf.while_loop(
            cond, body, loop_vars=[init_output_array, 0])

        return tf.cond(
            num_writes > 0,
            output_array.concat,
            lambda: tf.zeros(shape=[0], dtype=dtype),
            name=scope)


def repeat(elems, repeats):
    # get maximum repeat length in x
    maxlen = tf.reduce_max(repeats)

    # get the length of x
    xlen = tf.shape(repeats)[0]

    # create a range with the length of x
    # rng = tf.range(xlen)

    # tile it to the maximum repeat length, it should be of shape [xlen, maxlen] now
    rng_tiled = tf.tile(tf.expand_dims(elems, 1), tf.stack([1, maxlen], axis=0))

    # create a sequence mask using x
    # this will create a boolean matrix of shape [xlen, maxlen]
    # where result[i,j] is true if j < x[i].
    mask = tf.sequence_mask(repeats, maxlen)

    # mask the elements based on the sequence mask
    return tf.boolean_mask(rng_tiled, mask)


tf.InteractiveSession()

indices = np.random.random_integers(0, 20, [10])
repeats = np.random.random_integers(0, 4, [10])

# indices = [3, 6, 0]
# repeats = [2, 1, 2]

t0 = time.time()
print(repeat_loop(indices, repeats).eval())
t1 = time.time()
print(t1 - t0)
t0 = time.time()
print(repeat(indices, repeats).eval())
t1 = time.time()
print(t1 - t0)

# these are not the correct times though, I should use the proper runtime stats to measure the ops

"""
total_counts = 5


base_rep = repeat(indices, repeats)
count_rep = repeat(repeats, repeats)

maxlen = tf.reduce_max(repeats)
print(maxlen.eval())
coor_inc = tf.tile(tf.range(maxlen), multiples=tf.expand_dims(tf.cast(tf.ceil(total_counts / maxlen), tf.int64), -1))
# crop coor_inc
coor_inc = coor_inc[0:total_counts]
print(coor_inc.eval())
print(count_rep.eval())
print(base_rep.eval())

coors = base_rep + coor_inc % count_rep
print(coors.eval())
"""