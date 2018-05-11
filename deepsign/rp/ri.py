import random

import numpy as np


class RandomIndex:
    def __init__(self, dim, positive, negative):
        self.positive = positive
        self.negative = negative
        self.dim = dim
        self.s = len(positive) + len(negative)

    def sorted_indices_values(self):
        """Returns two lists indices, values with the indices
        and respective positive and negative values ordered by index"""
        indices = self.positive + self.negative
        values = [1] * len(self.positive) + [-1] * len(self.negative)
        indices, values = zip(*[(i, v) for i, v in sorted(zip(indices, values), key=lambda pair: pair[0])])

        return list(indices), list(values)

    def to_vector(self):
        v = np.zeros(self.dim)
        v[self.positive] = 1
        v[self.negative] = -1
        return v

    def to_dist_vector(self):
        """
        Returns a vector of dimension dim*2 with the sparse distribution for the positive
        and negative labels concatenated [pos dist][neg dist]

        """
        v = np.zeros(self.dim * 2)
        v[self.positive] = 1 / self.s
        negative = np.array(self.negative) + self.dim
        v[negative] = 1 / self.s
        return v

    def to_class_vector(self):
        """Same as dist vector but returns a binary version of all active classes"""
        v = np.zeros(self.dim * 2)
        v[self.positive] = 1
        negative = np.array(self.negative) + self.dim
        v[negative] = 1
        return v

    def to_class_indices(self):
        """ Same as to_class_vector but returns the final indices only
        """
        positive = np.array(self.positive)
        negative = np.array(self.negative) + self.dim

        return np.concatenate([positive, negative])

    def get_positive_vector(self):
        v = np.zeros(self.dim)
        v[self.positive] = 1
        return v

    def get_negative_vector(self):
        v = np.zeros(self.dim)
        v[self.negative] = 1
        return v

    def __str__(self):
        return "RI" + str((self.dim, self.s)) + ":\n\t+1 = " + str(self.positive) + "\n\t-1 = " + str(self.negative)


def ri_from_indexes(dim, active_indexes):
    """
    Creates a random index instance from a given list of active indexes. It assumes the following:
        -the first half of the indexes corresponds to the positive entries
        -the second half of the indexes corresponds to the negative entries
        -the vales of active indexes are >0 and <dim

    :param dim: the dimension of the random index to be created
    :param n_active: number of active elements
    :param active_indexes: list of active indexes
    :return: a new random index instance
    """
    n_active = len(active_indexes)

    num_positive = n_active // 2

    positive = active_indexes[0:num_positive]
    negative = active_indexes[num_positive:len(active_indexes)]

    return RandomIndex(dim=dim, positive=positive, negative=negative)


class Generator:
    """ A random-state-preserving random index generator.

    Generates ternary random index instances which can then be converted to sparse vectors.
    The instances only save information about the dimension of the indexes (vectors), and
    their positive and negative indexes.
    """

    def __init__(self, dim, num_active, seed=None, symmetric=True):
        # system time is used if seed not supplied
        if seed:
            random.seed(seed)

        self.dim = dim
        self.num_active = int(num_active)
        self.symmetric = symmetric

    def generate(self):
        # ensure that you can make other calls to random
        # random.setstate(self.random_state)

        active_indexes = random.sample(range(self.dim), self.num_active)

        if self.symmetric:
            num_positive = self.num_active // 2
            positive = active_indexes[0:num_positive]
            negative = active_indexes[num_positive:len(active_indexes)]
        else:
            positive = active_indexes
            negative = []

        # ensure that you can make other calls to random
        # self.random_state = random.getstate()

        return RandomIndex(self.dim, positive, negative)

    def generate_v2(self):
        active_indexes = np.random.choice(np.arange(self.dim), self.num_active, replace=False)
        positive, negative = np.split(active_indexes, 2)

        return RandomIndex(self.dim, positive, negative)
