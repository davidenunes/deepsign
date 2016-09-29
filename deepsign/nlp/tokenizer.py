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
        text = self.text

        # skip space characters
        if rem.match(pm.RE_SPACES, text):
            self.text = rem.skip()
            return self.__next__()

        # CONTRACTIONS *************************************************************************************************
        # contractions resulted from previous separation of contraction words
        # (do)(n't) -> return (do) ->  skip to (n't)
        if rem.match(pm.CONTRACTION, text):
            self.text = rem.skip()
            return rem.matched.group()

        # he's don't I'ven't -> (w)(c+)
        if rem.match(pm.CONTRACTION_WORD_1, text):
            self.text = rem.skip(1)
            return rem.matched.group(1)

        # 'twas 'tis -> (c)(w)
        if rem.match(pm.CONTRACTION_WORD_2, text):
            self.text = rem.skip(1)
            return rem.matched.group(1)

        # y'all -> (c)(w)
        if rem.match(pm.CONTRACTION_WORD_3, text):
            self.text = rem.skip(1)
            return rem.matched.group(1)

        # (w) -> (w)
        if rem.match(pm.CONTRACTION_WORD_EXTRA, text):
            self.text = rem.skip()
            return rem.matched.group()

        # WORDS ********************************************************************************************************
        """
        Check for entities that might start with words before splitting based on words
        e.g.
        """
        if rem.match(pm.WORD, text):
            self.text = rem.skip(0)
            return rem.matched.group()

        # NUMBERS ******************************************************************************************************
        # TODO: requisites not met
        """
        Check for entities that might start with numbers before splitting based on numbers
        e.g. times, dates, etc
        """
        if rem.match(pm.NUMBER, text):
            self.text = rem.skip(0)
            return rem.matched.group()

        # PUNCTUATION **************************************************************************************************
        # TODO: requisites not met
        """
        We should check for all entities that might start with punctuation before checking this
        e.g.
        - numbers with sign (-2.5)
        - hashtags
        - twitter handles
        - arrows
        - emoticons
        """
        if rem.match(pm.PUNCT_SEQ, text):
            self.text = rem.skip(0)
            return rem.matched.group()


        # skip everything else character by character
        else:
            token = self.text[0]
            self.text = self.text[1:]
            return token


def tokenize(text):
    tk = Tokenizer(text)
    return [token for token in tk]