import itertools


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def divide_slice(num_elems,num_slices=1):
    """ Splits an vector with num_elems into num_slices returning a list of
    slices for that vector

    :param num_elems: number of elemebts in the vector
    :param num_slices: number of slices the vector is to be splitted into
    :return: a list of slices for the vector
    """
    len_split = int(num_elems / num_slices)
    num_indexes = num_slices - 1

    ss = [0]
    for s in range(len_split, len_split * num_slices, len_split):
        ss.append(s)

    ss.append(num_elems)
    slices = [slice(s[0],s[1],1) for s in _pairwise(ss)]

    return slices



