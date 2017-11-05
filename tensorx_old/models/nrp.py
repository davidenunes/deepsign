
import tensorflow as tf
from tensorx_old.init import glorot_init, random_uniform_init
from tensorx_old.layers import Input, Dense, Act, Embeddings, SparseInput


class NRPDAE:
    """ Neural Random Projections De-noising Auto-Encoder
    
    """
    def __init__(self,
                 k_dim,                         # random projection dimension
                 n_active,                      # number of active features
                 h_dim,                         # hidden layer size
                 h_init=random_uniform_init,    # embedding initialisation
                 h_act=Act.elu):                # hidden layer activation

        self.k_dim = k_dim
        self.n_active = n_active

        num_pos = n_active / 2
        num_neg = n_active - num_pos

        self.pos_input = SparseInput(n_units=k_dim, n_active=num_pos, dtype=tf.int32, name="positive features")
        self.neg_input = SparseInput(n_units=k_dim, n_active=num_neg, dtype=tf.int32, name="negative features")







