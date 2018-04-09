import tensorflow as tf


def enum_each(enum_sizes, name="repeat_each"):
    """ creates an enumeration for each repeat
    and concatenates the results because we can't have
    a tensor with different row or column sizes

    Example:

        enum_each([1,2,4])

        Returns

        [0,0,1,0,1,2,3]

        the enums are [0], [0,1], [0,1,2,3]

    Args:
        x: Tensor with the same shape as repeats
        enum_sizes: Tensor with the same shape as x
        name: name for this op

    Returns:

    """
    with tf.name_scope(name, values=[enum_sizes]):
        enum_sizes = tf.convert_to_tensor(enum_sizes)

        # get maximum repeat length in x
        maxlen = tf.reduce_max(enum_sizes)

        x = tf.range(maxlen)

        # tile it to the maximum repeat length, it should be of shape [maxlen x maxlen] now
        x_repeat = tf.stack([maxlen, 1], axis=0)
        x_tiled = tf.tile(tf.expand_dims(x, 0), x_repeat)

        # create a sequence mask using x
        # this will create a boolean matrix of shape [xlen, maxlen]
        # where result[i,j] is true if j < x[i].
        mask = tf.sequence_mask(enum_sizes, maxlen)

        # mask the elements based on the sequence mask
        return tf.boolean_mask(x_tiled, mask)



tf.InteractiveSession()

counts = [0, 3, 1, 2]
print(enum_each(counts).eval())
