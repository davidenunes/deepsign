#!/usr/bin/env python
"""
RexEx Tokenizer
"""

from deepsign.nlp import patterns as pm
import re


RE = {
    'SPACES': re.compile(pm.SPACES),
    'CONTRACTION': re.compile(pm.CONTRACTION),
    'CONTRACTION_W1': re.compile(pm.CONTRACTION_WORD_1),
    'CONTRACTION_W2': re.compile(pm.CONTRACTION_WORD_2),
    'CONTRACTION_W3': re.compile(pm.CONTRACTION_WORD_3),
    'CONTRACTION_WE': re.compile(pm.CONTRACTION_WORD_EXTRA),
    'WORD': re.compile(pm.WORD),
    'NUMBER': re.compile(pm.NUMBER),
    'PUNCT_SEQ': re.compile(pm.PUNCT_SEQ)
}

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
        matcher = pm.REMatcher()
        text = self.text

        # return space characters or sequences of whitespace as a single token
        if matcher.match(RE['SPACES'], text):
            self.text = matcher.skip()
            return matcher.matched.group()

        # CONTRACTIONS *************************************************************************************************
        # contractions resulted from previous separation of contraction words
        # (do)(n't) -> return (do) ->  skip to (n't)
        if matcher.match(RE['CONTRACTION'], text):
            self.text = matcher.skip()
            return matcher.matched.group()


        if matcher.match(RE['CONTRACTION_W1'], text):
            # he's don't I'ven't -> (w)(c+)
            self.text = matcher.skip(1)
            return matcher.matched.group(1)

        # 'twas 'tis -> (c)(w)
        if matcher.match(RE['CONTRACTION_W2'], text):
            self.text = matcher.skip(1)
            return matcher.matched.group(1)

        # y'all -> (c)(w)
        if matcher.match(RE['CONTRACTION_W3'], text):
            self.text = matcher.skip(1)
            return matcher.matched.group(1)

        # words that we don't want to split but have an apostrophe (w) -> (w)
        if matcher.match(RE['CONTRACTION_WE'], text):
            self.text = matcher.skip()
            return matcher.matched.group()

        # TODO test abbreviations
        # TODO test conflict between versions and numbers with dot and coma
        # WORDS ********************************************************************************************************
        """
        Check for entities that might start with words before splitting based on words
        e.g.
        - URLs: something.com
        - e-mail: abc@xyz.com
        - Sensored words: f**k
        - Abbreviations: Ph.D a.k.a. p.m.
        - Versions: v1.2
        """
        if matcher.match(RE['WORD'], text):
            self.text = matcher.skip(0)
            return matcher.matched.group()

        # NUMBERS ******************************************************************************************************
        # TODO: requirements not met
        """
        Check for entities that might start with numbers before splitting based on numbers
        e.g.
        - times
        - dates
        - version numbers
        """
        if matcher.match(RE['NUMBER'], text):
            self.text = matcher.skip(0)
            return matcher.matched.group()

        # PUNCTUATION **************************************************************************************************
        # TODO: requirements not met
        """
        We should check for all entities that might start with punctuation before checking this
        e.g.
        - numbers with sign (-2.5)
        - hashtags
        - twitter handles
        - arrows
        - hearts
        - emoticons
        """
        if matcher.match(RE['PUNCT_SEQ'], text):
            self.text = matcher.skip(0)
            return matcher.matched.group()


        # skip everything else character by character
        else:
            token = self.text[0]
            self.text = self.text[1:]
            return token


def tokenize(text):
    tk = Tokenizer(text)
    return [token for token in tk]