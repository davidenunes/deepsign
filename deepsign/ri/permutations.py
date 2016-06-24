import numpy as np
import random


def to_matrix(perm_vector):
    """ Convert permutation vector to a permutation matrix
    that can be used directly to permute a given vector by multiplying
    the vector by the matrix

    :param perm_vector: a permutation vector.
    Each index contains the index to which it will be permuted
    :return: a permutation matrix
    """
    n = len(perm_vector)
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[perm_vector[i], i] = 1

    return matrix


def to_vector(perm_matrix):
    """ Convert a permutation matrix to a permutation vector
     where each index has the index to which the entry is to be permuted

    :param perm_matrix: the permutation matrix to be converted
    :return: a permutation vector
    """
    n = perm_matrix.shape[0]
    cols = range(n)

    perm_vector = [np.flatnonzero(perm_matrix[:, c])[0] for c in cols]
    return perm_vector


def permute_sparse(sparse_vector, perm_vector):
    """Permutes a sparse vector by applying only the permutations on the active (non-zero)
    elements of the vector.

    :param sparse_vector: a vector with fez non-zero entries (as compared with dimension)
    :param perm_vector: a vector where each entry contains the index to which the entry in the current
    index is to be permuted

    :return: a new (copy) of the vector with the permuted entries
    """
    nz = np.flatnonzero(sparse_vector)

    permuted_v = np.copy(sparse_vector)
    permuted_v[nz] = permuted_v[perm_vector[nz]]

    return permuted_v


def permute(vector,permutation):
    """ Permutes a given vector by multiplying it by its permutation matrix;
    this doesn't make any assumptions about the sparsity of the given vector

    :param vector: the vector to be permuted
    :param permutation: the permutation (if its not a matrix we assume its a vector and try to convert it)
    :return: a new vector resulting from the permutation
    """
    p = permutation
    if not isinstance(permutation, np.matrix):
        p = to_matrix(p)

    pvector = np.dot(vector,p)
    return pvector


class PermutationGenerator:
    """A random-state-preserving permutation generator.

    Generates permutation matrices and vectors and preserves
    the state of the random number generator while doing so. This is just to ensure that the order in which we execute
    the code doesn't matter. Also we can assign a seed to the generator and generate permutations from that seed
    while preserving the state of the random number generator (even if random is used throughout the program, this will
    not affect the permutation generation)

    """
    def __init__(self, dim, seed=None):
        # system time is used if seed not supplied
        if seed:
            random.seed(seed)

        self.random_state = random.getstate()
        self.dim = dim

    def matrix(self, n):
        # ensure that you can make other calls to random
        random.setstate(self.random_state)

        indexes = list(range(n))
        random.shuffle(indexes)
        matrix = to_matrix(indexes)

        # ensure that you can make other calls to random
        self.random_state = random.getstate()

        return matrix

    def permutation_vector(self,n):
        # ensure that you can make other calls to random
        random.setstate(self.random_state)

        # ensure that you can make other calls to random
        self.random_state = random.getstate()
        return random.shuffle(range(n))