import tensorflow as tf
import os
import numpy as np


def save_variable(tf_session, tf_variable, out_dir, embeddings_name="embeddings"):
    """
    Saves a numpy variable to a file using the numpy format
    :param tf_session: the tensorflow session
    :param tf_variable: the variable to be saved
    :param out_dir: the output directory
    :param embeddings_name: the name to be given to the file (no extension)
    """
    w = tf_session.run(tf_variable)
    save_dir = os.path.dirname(out_dir)
    if embeddings_name is None:
        embeddings_name = "{}.npy".format(embeddings_name)

    embeddings_file = os.path.join(save_dir, embeddings_name)
    np.save(embeddings_file, w)


def save_model(tf_session, filename, step=None):
    """
    Saves the current model to the disk 
    :param tf_session: the current tensorflow session object
    :param filename: the complete filename (including the path) which dictates where the model should be saved
    :param step: current simulation step can be used to create different checkpoints
    """
    saver = tf.train.Saver()
    # saves in name.meta
    saver.save(tf_session, filename, global_step=step)


def load_model(session, filename):
    """
    Restores the model using the file from the disk. 
    Note that the graph has to be equal to that which was previously saved. If you want to save the current global step
    the best way is to store it in a variable which means this will be restored as well when this method is run
    
    :param session: 
    :param filename: 
    """
    saver = tf.train.Saver()
    saver.restore(session, filename)


def save_graph(session,out_dir):
    """
    Saves the current graph to a file. This can be visualised running tensorboard on the same directory
    :param session: current tensorflow session
    :param out_dir: output directory
    """
    writer = tf.summary.FileWriter(out_dir,graph=session.graph)
    writer.flush()
