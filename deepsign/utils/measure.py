import numpy as np


def cosine(u,v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def angular_similarity(u,v):
    cosine_sim = cosine(u,v)
    angular_dist = np.arcos(cosine_sim) / np.pi
    return 1 - angular_dist



def gini(x):
    """
    Gini coefficient in relative mean difference form

    x: numpy array with positive entries

    Notes
    -----
    Reference: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    From: https://github.com/pysal/pysal/blob/master/pysal/inequality/gini.py
    """
    n = len(x)

    x_sum = x.sum()
    if x_sum == 0:
        return 0.

    n_x_sum = n * x_sum
    r_x = (2. * np.arange(1, len(x)+1) * x[np.argsort(x)]).sum()
    return (r_x - n_x_sum - x_sum) / n_x_sum