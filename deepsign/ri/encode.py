import deepsign.text.windows as txtw
import deepsign.ri.permutations as perm
import numpy as np


class Encoder:
    def encode(self,seq=[]):
        return seq


class WindowEncoder(Encoder):
    def __init__(self, sign_index, window_size=1):
        self.window_size = window_size
        self.sign_index = sign_index


class BoW(WindowEncoder):
    def __init__(self, sign_index, window_size=1):
        super(BoW, self).__init__(sign_index, window_size)

    def encode(self,seq=[]):
        """encodes a sequence of tokens using a slidding window encoder
        with or without including the target word of each window

        :param seq: ['hello', 'this', 'is my sequence'] a sequence of tokens to be encoded using random indexing
        :return:
        """
        index = self.sign_index

        def get_vectors(window):
            left_v = [index.get_ri(s).to_vector() for s in window.left]
            right_v = [index.get_ri(s).to_vector() for s in window.right]
            target_v = [index.get_ri(window.word).to_vector()]

            return left_v + right_v + target_v

        windows = txtw.sliding_windows(seq,self.window_size)
        vectors = [np.sum(get_vectors(w),axis=0) for w in windows]

        return vectors


class BoWDir(WindowEncoder):
    def __init__(self, sign_index, window_size=1):
        super(BoWDir, self).__init__(sign_index, window_size)

        # create and store new permutation to be used as order information
        dim = self.sign_index.feature_dim()
        gen = perm.PermutationGenerator(dim=dim)

        self.left_permutation = gen.matrix()
        self.right_permutation = np.linalg.inv(self.left_permutation)

    def encode(self,seq=[]):
        index = self.sign_index

        windows = txtw.sliding_windows(seq, self.window_size)

        # use +1 to represent right and -1 to represent left

        vectors = []

        for w in windows:

            left_vs = [index.get_ri(s).to_vector() for s in w.left]
            right_vs = [index.get_ri(s).to_vector() for s in w.right]

            left_v = np.zeros(index.feature_dim())
            right_v = np.zeros(index.feature_dim())

            if len(left_vs) > 0:
                left_v = np.sum(left_vs,axis=0)
                left_v = np.dot(left_v,self.left_permutation)
            if len(right_vs) > 0:
                right_v = np.sum(right_vs, axis=0)
                right_v = np.dot(right_v,self.right_permutation)

            target_v = index.get_ri(w.word).to_vector()

            result_v = left_v + target_v + right_v
            vectors.append(result_v)

        return vectors