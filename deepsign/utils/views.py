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


class Window():
    """ A window contains:
        a left []
        a target which is in the center of the window
        a right []
    """
    def __init__(self, left, target, right):
        self.left = left
        self.target = target
        self.right = right

    def __str__(self):
        return "("+str(self.left)+","+self.target + "," + str(self.right) + ")"


def sliding_windows(seq, window_size=1):
    """ converts an array of strings to a sequence of windows

    :param seq: a sequence to be sliced into windows
    :param window_size: the size of the window around each element
    :return: an array of Window instances
    """
    elem_indexes = range(0, len(seq))
    num_elems = len(seq)

    windows = []
    # create a sliding window for each elem
    for w in elem_indexes:

        # lower limits
        wl = max(0,w - window_size)
        wcl = w

        # upper limits
        wch = num_elems if w == num_elems-1 else min(w+1, num_elems-1)
        wh = w + min(window_size + 1, num_elems)

        # create window
        left = seq[wl:wcl]
        target = seq[w]
        right = seq[wch:wh]

        windows.append(Window(left, target, right))

    return windows
