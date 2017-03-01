from bidict import bidict
import marisa_trie
import numpy as np
import os
import h5py

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

    def contains_id(self, sign_id):
        return id in self.signs.inv

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
    @staticmethod
    def map_frequencies(vocabulary, frequencies, sign_trie):
        """creates a new dict mapping ids from sign_trie to the frequencies of the words
        in a given vocabulary (also stored in the given sign_trie_index

        :param vocab: a list of words
        :param freq: a list of frequencies for the given words
        :param sign_trie: a sign index with all the given words
        :return:
        """
        freq = {}
        for i in range(len(sign_trie)):
            w = vocabulary[i]
            f = frequencies[i]
            freq[sign_trie.get_id(w)] = f

        return freq

    def __init__(self, generator, vocabulary=[], pregen_indexes=False):
        """
        :param generator a random index generator
        :param vocabulary an iterable of signs to be added
        :param frequencies an iterable with expected lookup frequencies
        :param pregen_indexes if true generates all the indexes for all the signs
        """
        self.generator = generator
        n_signs = len(vocabulary)

        self.sign_trie = marisa_trie.Trie(vocabulary)
        self.pregen_indexes = pregen_indexes
        if pregen_indexes:
            self.random_indexes = {i: generator.generate() for i in range(n_signs)}
        else:
            self.random_indexes = dict()

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
            n_prev_ids = len(self.sign_trie)

            # the key ids are preserved
            signs = old_signs + new_signs

            # build the new trie
            self.sign_trie = marisa_trie.Trie(signs)

    def remove(self, sign):
        """Removes a sign from the index

        Remove is expensive because marisa trie does not support efficient removal.
        Removing signs creates a new trie from scratch.

        :param sign: the sign to be removed
        """
        if sign in self.sign_trie:
            sid = self.sign_trie[sign]
            new_signs = [s for s in self.sign_trie.keys() if s != sign]
            # update trie
            self.sign_trie = marisa_trie.Trie(new_signs)
            if sid in self.random_indexes:
                self.random_indexes.pop(sign)

    def get_ri(self, sign):
        v = None
        if sign in self.sign_trie:
            sid = self.sign_trie[sign]

            if not self.pregen_indexes and sid not in self.random_indexes:
                self.random_indexes[sid] = self.generator.generate()
            v = self.random_indexes[sid]
        return v

    def get_id(self, sign):
        sign_id = None
        if sign in self.sign_trie:
            sign_id = self.sign_trie[sign]
        return sign_id

    def get_sign(self, sign_id):
        try:
            return self.sign_trie.restore_key(sign_id)
        except KeyError:
            return None

    def contains_id(self, sign_id):
        return sign_id < len(self.sign_trie)

    def save(self, filename, dir):
        """
        Saves the current index to an hdf5 store
        :param filename: the filename without extension that will be used
        to store the index, uses hdf5 as extension
        """
        output_dir = dir
        if not os.path.exists(output_dir):
            output_dir = os.getcwd()
        output_file = output_dir + "/" + filename + ".hdf5"
        h5index = h5py.File(output_file, 'w')

        keys = self.sign_trie.items()
        ws, ids = zip(*keys)

        # save signs
        dt = h5py.special_dtype(vlen=str)
        h5index.create_dataset("signs", data=np.array(ws), dtype=dt, compression="gzip")

        # save random indexes
        if len(self.random_indexes) == 0:
            self.random_indexes = {i: self.generator.generate() for i in range(len(self))}

        ris = [self.random_indexes[id] for id in ids]
        ris = np.array([ri.positive + ri.negative for ri in ris])

        ri_dataset = h5index.create_dataset("ri", data=ris, compression="gzip")
        ri_dataset.attrs["k"] = self.generator.dim
        ri_dataset.attrs["s"] = self.generator.num_active

        h5index.close()

        #ri_vectors = [sign_index.get_ri(w.decode("UTF-8")) for w in word_ids]
        # store random indexes as a list of positive indexes followed by negative indexes
        #ri_sparse = np.array([ri.positive + ri.negative for ri in ri_vectors])
        #index_data = output_hdf5.create_dataset(dataset_name, data=ri_sparse, compression="gzip")
        #print("random index vectors written")
        #index_data.attrs["dimension"] = ri_dim
        #index_data.attrs["active"] = ri_active







