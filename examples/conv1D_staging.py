import tensorflow as tf
import tensorx as tx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sess = tf.InteractiveSession()

num_filters = 2
input_dim = 2
seq_size = 3
batch_size = 1

kernel_size = 2
filter_shape = [kernel_size, input_dim, num_filters]

x_concat = tf.ones([batch_size, input_dim * seq_size])

"""
N: number of images in the batch
H: height of the image
W: width of the image
C: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)

since we're processing vector representations

N == batch_size == 2
H == 1 (we're working with vectors)
W == input_dim == 2
C == channels == 1

NWC, channels are last we only use one
"""

x = tf.reshape(x_concat, [batch_size, seq_size, input_dim])

x_layer = tx.TensorLayer(x, input_dim)

print(x.eval())
print(x_layer.tensor)

filters = tf.get_variable("filters",
                          shape=filter_shape,
                          dtype=tf.float32,
                          initializer=tf.initializers.random_uniform(-1, 1))

filters = tf.ones(filter_shape)

c_layer = tx.Conv1D(x_layer, num_filters, kernel_size, shared_filters=filters, dilation_rate=1)


causal_layer = tx.CausalConv(x_layer, num_filters, kernel_size, shared_filters=filters, dilation_rate=2)

tf.global_variables_initializer().run()
print("Filters \n", filters.eval())

c = tf.nn.conv1d(x, filters, stride=1, padding="SAME", use_cudnn_on_gpu=True, data_format="NWC")
# c_concat = tf.nn.conv1d(x_concat, filters, stride=1, padding="SAME", use_cudnn_on_gpu=True, data_format="NWC")

print("conv_out_shape: ",c_layer.shape)
print("filter_shape: ", c_layer.filter_shape)
print("conv 1d \n", c.eval())
print("conv layer \n", c_layer.tensor.eval())
print("causal layer \n", causal_layer.tensor.eval())
# print("Result Concat \n", c_concat.eval())
