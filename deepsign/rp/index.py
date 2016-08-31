from bidict import bidict
import os


class SignIndex():

    def __init__(self, generator):
        """
        :param generator a random index generator
        """
        self.generator = generator
        self.signs = bidict() # str <-> id
        self.random_indexes = dict() # id -> RandomIndex
        self.nextID = 0

    def size(self):
        return len(self.signs)

    def feature_dim(self):
        return self.generator.dim

    def feature_act(self):
        return self.generator.num_active

    def contains(self,sign):
        return sign in self.signs

    def add(self,sign):
        #print("index PID ",os.getpid())
        if not sign in self.signs:
            # 1 - get next id
            self.signs[sign] = self.nextID
            # 2 . generate random index
            self.random_indexes[self.nextID] = self.generator.generate()
            self.nextID += 1

    def add_all(self,signs):
        for sign in signs:
            self.add(sign)

    def remove(self,sign):
        if sign in self.signs:
            del self.signs[sign]

    def get_ri(self,sign):
        id = self.signs[sign]

        v = None
        if id is not None:
            v = self.random_indexes[id]

        return v

    def get_id(self,sign):
        id = None
        if sign in self.signs:
            id = self.signs[sign]
        return id

    def get_sign(self,id):
        sign = None
        if id in self.signs.inv:
            sign = self.signs.inv[id]
        return sign
