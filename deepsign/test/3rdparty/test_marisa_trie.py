from unittest import TestCase
import marisa_trie


class TestMarisaTrie(TestCase):
    def test_id_inc(self):
        """Ids start at 0 and follow the order of the keys"""
        keys = ["a", "b", "c"]
        trie = marisa_trie.Trie(keys)

        for i in range(len(keys)):
            self.assertEqual(i, trie[keys[i]])