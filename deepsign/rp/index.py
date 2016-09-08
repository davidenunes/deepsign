from bidict import bidict
import os


class SignIndex:
    def __init__(self, generator):
        """
        :param generator a random index generator
        """
        self.generator = generator
        self.signs = bidict()  # str <-> id
        self.random_indexes = dict()  # id -> RandomIndex
        self.nextID = 0

    def __len__(self):
        return len(self.signs)

    def feature_dim(self):
        return self.generator.dim

    def feature_act(self):
        return self.generator.num_active

    def contains(self, sign):
        return sign in self.signs

    def add(self, sign):
        if sign not in self.signs:
            # 1 - get next id
            sign_id = self.nextID
            self.signs[sign] = sign_id
            # 2 . generate random index
            self.random_indexes[sign_id] = self.generator.generate()
            self.nextID += 1

    def add_all(self, signs):
        for sign in signs:
            self.add(sign)

    def remove(self, sign):
        if sign in self.signs:
            del self.signs[sign]

    def get_ri(self, sign):
        id = self.signs[sign]

        v = None
        if id is not None:
            v = self.random_indexes[id]

        return v

    def get_id(self, sign):
        sign_id = None
        if sign in self.signs:
            sign_id = self.signs[sign]
        return sign_id

    def get_sign(self, sign_id):
        sign = None
        if sign_id in self.signs.inv:
            sign = self.signs.inv[sign_id]
        return sign
