import tensorflow as tf
import tensorx as tx

sess = tf.InteractiveSession()

num_filters = 2
input_dim = 2
seq_size = 3
batch_size = 2

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

print(x.eval())

filters = tf.get_variable("filters",
                          shape=filter_shape,
                          dtype=tf.float32,
                          initializer=tf.initializers.random_uniform(-1, 1))

tf.global_variables_initializer().run()
print("Filters \n", filters.eval())

c = tf.nn.conv1d(x, filters, stride=1, padding="SAME", use_cudnn_on_gpu=True, data_format="NWC")
# c_concat = tf.nn.conv1d(x_concat, filters, stride=1, padding="SAME", use_cudnn_on_gpu=True, data_format="NWC")

print("Result \n", c.eval())
print("Result Concat \n", c_concat.eval())
