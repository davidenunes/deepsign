import os
import itertools


class WikiTextIterator:
    """
    Simple iterator, the file since the file has one sentence per line
    """

    def __init__(self, path, max_samples=None, mark_eos=False):
        assert os.path.exists(path)
        self.path = path
        self.current_sentence = None
        self.source = open(path, 'r', encoding="utf-8")

        self.mark_eos = mark_eos
        self.max_samples = max_samples
        self.num_samples = 0

    def close(self):
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.max_samples is not None and self.num_samples >= self.max_samples:
            self.source.close()
            raise StopIteration

        self.current_sentence = self.source.readline()

        if not self.current_sentence:
            self.source.close()
            raise StopIteration

        tokens = self.current_sentence.split()
        # skip empty lines
        if len(tokens) == 0:
            return self.__next__()
        else:
            self.num_samples += 1
            if self.mark_eos:
                tokens.append("<eos>")
            return tokens


class WikiText103:
    UNKNOWN_TOKEN = "<unk>"

    """ 
            
        Args:
            path: path to the directory containing the dataset files
    """

    def __init__(self, path, mark_eos=False):
        self.mark_eos = mark_eos
        self.train_file = os.path.join(path, 'wiki.train.tokens')
        self.valid_file = os.path.join(path, 'wiki.valid.tokens')
        self.test_file = os.path.join(path, 'wiki.test.tokens')

        if not os.path.exists(self.train_file):
            raise FileNotFoundError("could find train set in {path}".format(path=self.train_file))
        if not os.path.exists(self.valid_file):
            raise FileNotFoundError("could find validation set in {path}".format(path=self.valid_file))
        if not os.path.exists(self.test_file):
            raise FileNotFoundError("could find test set in {path}".format(path=self.test_file))

    def training_set(self, n_samples=None):
        """
        :param n_samples: max number of sentences
        """
        return WikiTextIterator(path=self.train_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def validation_set(self, n_samples=None):
        return WikiTextIterator(path=self.valid_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def test_set(self, n_samples=None):
        return WikiTextIterator(path=self.test_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def full(self):
        return itertools.chain(self.training_set(),
                               self.validation_set(),
                               self.test_set())
