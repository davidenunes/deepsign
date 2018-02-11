# Start a TensorFlow server as a single-process "cluster".
import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)


c = tf.constant("Hello, distributed TensorFlow!")

server = tf.train.Server.create_local_server()
print(server.target)

# Create a session on the server.
sess = tf.Session(server.target)
s = sess.run(c)
print(s)
sess.close()
