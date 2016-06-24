import deepsign.text.windows as txtw


class Encoder:
    def encode(self,seq=[]):
        return seq


class BoWEncoder(Encoder):
    def __init__(self,sign_index, window_size=1,include_target=False):
        self.window_size = window_size
        self.sign_index = sign_index
        self.include_target = include_target

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
            target_v = []
            if self.include_target:
                target_v = [index.get_ri(window.word).to_vector()]

            return left_v + right_v + target_v

        windows = txtw.sliding_windows(seq,self.window_size)
        vectors = [sum(get_vectors(w)) for w in windows]

        return vectors

