import numpy as np
from numpy import linalg as LA


def to_bow(window, sign_index):
    left_vs = [sign_index.get_ri(s).to_vector() for s in window.left]
    right_vs = [sign_index.get_ri(s).to_vector() for s in window.right]
    target_v = [sign_index.get_ri(window.target).to_vector()]

    ri_vs = left_vs + target_v + right_vs
    bow_v = np.sum(ri_vs, axis=0)

    return bow_v

def to_bow_dir(window, sign_index, perm_matrix):
    target_v = sign_index.get_ri(window.target).to_vector()
    bow_dir_v = target_v

    # permutations for direction encoding
    right_perm = perm_matrix
    left_perm = LA.inv(perm_matrix)

    left_vs = [sign_index.get_ri(s).to_vector() for s in window.left]
    right_vs = [sign_index.get_ri(s).to_vector() for s in window.right]

    if len(left_vs) > 0:
        left_v = np.sum(left_vs, axis=0)
        left_v = np.dot(left_v, left_perm)
        bow_dir_v += left_v
    if len(right_vs) > 0:
        right_v = np.sum(right_vs, axis=0)
        right_v = np.dot(right_v, right_perm)
        bow_dir_v += right_v

    return bow_dir_v


def to_bow_order(window, sign_index, perm_matrix):
    target_v = sign_index.get_ri(window.target).to_vector()
    bow_order_v = target_v

    left_vs = [sign_index.get_ri(s).to_vector() for s in window.left]
    right_vs = [sign_index.get_ri(s).to_vector() for s in window.right]

    if len(left_vs) > 0:
        # use negative power permutation to encode left of window
        for i, lv in enumerate(reversed(left_vs)):
            distance_i = -1 * (i + 1)
            perm_i = LA.matrix_power(perm_matrix, distance_i)
            lv_perm = np.dot(lv, perm_i)
            bow_order_v += lv_perm

    if len(right_vs) > 0:
        # positive power permutations to encode right of window
        for i, rv in enumerate(right_vs):
            distance_i = (i + 1)
            perm_i = LA.matrix_power(perm_matrix, distance_i)
            rv_perm = np.dot(rv, perm_i)
            bow_order_v += rv_perm

    return bow_order_v


