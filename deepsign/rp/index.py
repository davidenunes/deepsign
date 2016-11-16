from bidict import bidict
import marisa_trie
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


class TrieSignIndex:
    def __init__(self, generator, signs):
        """
        :param generator a random index generator
        """
        self.generator = generator
        self.sign_trie = marisa_trie.Trie(signs)
        self.random_indexes = {i: generator.generate() for i in range(len(signs))}

    def __len__(self):
        return len(self.sign_trie)

    def feature_dim(self):
        return self.generator.dim

    def feature_act(self):
        return self.generator.num_active

    def contains(self, sign):
        return sign in self.sign_trie

    def add(self, sign):
        """ Adding a single element to a new TrieSignIndex is quite expensive since we have to rebuild the index
            add chunks at a time for a better performance
        """
        self.add_all([sign])

    def add_all(self, signs):
        new_signs = [s for s in signs if s not in self.sign_trie]
        if len(new_signs) > 0:
            old_signs = self.sign_trie.keys()
            # the key ids are preserved
            signs = old_signs + new_signs

            # build the new trie
            self.sign_trie = marisa_trie.Trie(signs)

            # generate more ris
            n_prev_ids = len(self.sign_trie)
            new_ids = range(n_prev_ids, n_prev_ids + len(new_signs))

            new_ri = {i: self.generator.generate() for i in new_ids}
            self.random_indexes.update(new_ri)


    def remove(self, sign):
        raise Exception("Operation not supported by TrieSignIndex")

    def get_ri(self, sign):
        v = None
        if sign in self.sign_trie:
            sid = self.sign_trie[sign]
            v = self.random_indexes[sid]
        return v

    def get_id(self, sign):
        sign_id = None
        if sign in self.sign_trie:
            sign_id = self.sign_trie[sign]
        return sign_id

    def get_sign(self, sign_id):
        sign = None

        try:
            self.sign_trie.restore_key(sign_id)
        except KeyError:
            pass

        return sign


