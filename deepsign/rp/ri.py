import random

import numpy as np


class RandomIndex:
    def __init__(self, dim, positive, negative):
        self.positive = positive
        self.negative = negative
        self.dim = dim

    def to_vector(self):
        v = np.zeros(self.dim)
        v[self.positive] = 1
        v[self.negative] = -1
        return v


def ri_from_indexes(dim, n_active, active_indexes):
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
    num_positive = n_active // 2

    positive = active_indexes[0:num_positive]
    negative = active_indexes[num_positive:len(active_indexes)]

    return RandomIndex(dim=dim, positive=positive, negative=negative)


class RandomIndexGenerator:
    """ A random-state-preserving random index generator.

    Generates ternary random index instances which can then be converted to sparse vectors.
    The instances only save information about the dimension of the indexes (vectors), and
    their positive and negative indexes.
    """

    def __init__(self, dim, active, seed=None):
        # system time is used if seed not supplied
        if seed:
            random.seed(seed)

        self.random_state = random.getstate()
        self.dim = dim
        self.num_active = active

    def generate(self):
        # ensure that you can make other calls to random
        random.setstate(self.random_state)

        active_indexes = random.sample(range(0, self.dim), self.num_active)

        num_positive = self.num_active // 2
        positive = active_indexes[0:num_positive]
        negative = active_indexes[num_positive:len(active_indexes)]

        # ensure that you can make other calls to random
        self.random_state = random.getstate()

        return RandomIndex(self.dim, positive, negative)
