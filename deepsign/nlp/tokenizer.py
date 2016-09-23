#!/usr/bin/env python
"""
RexEx Tokenizer
"""

from collections import deque
from deepsign.nlp import patterns as pm


class Tokenizer:
    def __init__(self, text):
        self.text = text

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.text) == 0:
            raise StopIteration
        else:
            return self._next_token()

    def _next_token(self):
        rem = pm.REMatcher()
        s = self.text

        # skip space characters
        if rem.match(pm.RE_SPACES, s):
            self.text = rem.skip()
            return self.__next__()

        if rem.match(pm.RE_WORD, s):
            self.text = rem.skip()
            return rem.matched.group()

        else:#TODO remove this, finish the tokenizer
            token = self.text[0]
            self.text = self.text[1:]
            return token



def tokenize(text):
    tk = Tokenizer(text)
    return [token for token in tk]