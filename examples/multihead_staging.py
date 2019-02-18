import tensorflow as tf
import numpy as np
import tensorx as tx
from tensorx import Linear
import os
import six

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
    Raises:
    ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
    Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def mask(inputs, queries=None, keys=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)
    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                     [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                              [1., 0.]]], tf.float32)
    >> mask(inputs, queries, keys, "query")
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs * masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


sess = tf.InteractiveSession()

n_features = 3
embed_size = 128
seq_size = 3
batch_size = 2
n_heads = 8
h_dim = 512
causality = False

inputs = tx.TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
lookup = tx.Lookup(inputs, seq_size=seq_size, lookup_shape=[n_features, embed_size])

x = lookup
input_dim = lookup.n_units
# input_dim = input_layer.tensor.get_shape().as_list()[-1]

if h_dim % n_heads != 0:
    raise ValueError(
        "The hidden size {} is not a multiple of the number of attention "
        "heads {}".format(h_dim, n_heads))

wq = Linear(x, n_units=h_dim)  # (batch_size, steps, dim)
wk = Linear(x, n_units=h_dim)  # (batch_size, steps, dim)
wv = Linear(x, n_units=embed_size)  # (batch_size, steps, dim)

qh = tf.concat(tf.split(wq, n_heads, axis=2), axis=0)  # (h*batch_size, steps, dim/h)
kh = tf.concat(tf.split(wk, n_heads, axis=2), axis=0)  # (h*batch_size, steps, dim/h)
vh = tf.concat(tf.split(wv, n_heads, axis=2), axis=0)  # (h*batch_size, steps, dim/h)

# dot prod
dot_products = tf.matmul(qh, tf.transpose(kh, [0, 2, 1]))

# scale
dot_products /= input_dim ** 0.5

out = dot_products
# outputs = mask(dot_products, qh, kh, type="key")
# causality or future blinding masking
if causality:
    out = mask(out, type="future")

out = tf.nn.softmax(out)

# This is actually dropping out entire tokens to attend to, which might
# seem a bit unusual, but is taken from the original Transformer paper.
# ?????
# out = dropout(out, drop_probability)

attention = tf.transpose(out, [0, 2, 1])
# query masking?
# outputs = mask(outputs, qh, kh, type="query")

# weighted sum (context vectors)
out = tf.matmul(out, vh)  # (N, T_q, d_v)

out = tf.concat(tf.split(out, n_heads, axis=0), axis=2)  # (N, T_q, d_model)

# test
tf.global_variables_initializer().run()

print("Embed Size: {} \nSteps: {} \nHeads: {}\n".format(embed_size, seq_size, n_heads))
print("Input Shape: ", tf.shape(x).eval())
print("WQ Shape: ", tf.shape(wq).eval())
print("QH Shape: ", tf.shape(qh).eval())
print("QK^T Shape: ", tf.shape(dot_products).eval())
print("out Shape: ", tf.shape(out).eval())
