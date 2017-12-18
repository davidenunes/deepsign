import xml.etree.ElementTree as ET
import os


class BNCReader:
    """ BNC Corpus Reader

        Implements a sentence iterator over BNC corpus files.
        This allows for:
            Iterate over the corpus returning sentences in the form of lists of strings
    """

    def __init__(self, filename):
        self.root = ET.parse(filename)
        self.source = open(filename)
        self.sentence_nodes = self.root.iterfind('.//s')

    def __iter__(self):
        return self

    def close(self):
        self.source.close()

    def __next__(self):
        # raises iteration stop if it doesn't have a next
        current_sentence_node = next(self.sentence_nodes)
        sentence = [token.strip() for token in current_sentence_node.itertext()]

        return sentence


def file_walker(source_dir):
    """Iterates over xml files given a source directory (/Texts)
    doesn't do it by file name order
    adding sorted works but loads all file names
    """
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.endswith(".xml"):
                yield os.path.join(root, name)
