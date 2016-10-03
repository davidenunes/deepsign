#!/usr/bin/env python
"""
RexEx Tokenizer
"""
import deepsign.utils.regex
from deepsign.nlp import patterns as pm
import re


RE = {
    'SPACES': re.compile(pm.SPACES, re.UNICODE),
    'CONTRACTION': re.compile(pm.CONTRACTION, re.UNICODE),
    'CONTRACTION_W1': re.compile(pm.CONTRACTION_WORD_1, re.UNICODE),
    'CONTRACTION_W2': re.compile(pm.CONTRACTION_WORD_2, re.UNICODE),
    'CONTRACTION_W3': re.compile(pm.CONTRACTION_WORD_3, re.UNICODE),
    'CONTRACTION_WE': re.compile(pm.CONTRACTION_WORD_EXTRA, re.UNICODE),
    'URL': re.compile(pm.URL, re.UNICODE),
    'WORD': re.compile(pm.WORD, re.UNICODE),
    'NUMERIC': re.compile(pm.NUMERIC, re.UNICODE),
    'PUNCT_SEQ': re.compile(pm.PUNCT_SEQ, re.UNICODE)
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
        matcher = deepsign.utils.regex.REMatcher()
        text = self.text
        match_group = -1

        # 0 is the whole match
        # 1 the first group (if it exists)
        if matcher.match(RE['SPACES'], text):
            # spaces are tokens too
            match_group = 0
        elif matcher.match(RE['CONTRACTION'], text):
            # (do)(n't) -> return (do) ->  skip to (n't)
            match_group = 0
        elif matcher.match(RE['CONTRACTION_W1'], text):
            # he's don't I'ven't -> (w)(c+)
            match_group = 1
        elif matcher.match(RE['CONTRACTION_W2'], text):
            # 'twas 'tis -> (c)(w)
            match_group = 1
        elif matcher.match(RE['CONTRACTION_W3'], text):
            # y'all -> (c)(w)
            match_group = 1
        elif matcher.match(RE['CONTRACTION_WE'], text):
            # words with apostrophe (w) -> (w)
            match_group = 0
        elif matcher.match(RE['URL']):
            match_group = 0

        elif matcher.match(RE['WORD'], text):
            # TODO: Match tokens that might start with words
            """e.g.
                - URLs: something.com
                - e-mail: abc@xyz.com
                - Sensored words: f**k
                - Abbreviations: Ph.D a.k.a. p.m.
                - Versions: v1.2
            """
            # TODO test abbreviations
            # TODO test conflict between versions and numbers with dot and coma
            match_group = 0
        elif matcher.match(RE['NUMERIC'], text):
            match_group = 0
        elif matcher.match(RE['PUNCT_SEQ'], text):
            # TODO match entities that start with punctuation before this
            """e.g.
                - numbers with sign (-2.5)
                - hashtags
                - twitter handles
                - arrows
                - hearts
                - emoticons
            """
            match_group = 0

        if match_group >= 0:
            self.text = matcher.skip(match_group)
            token = matcher.matched.group(match_group)
        else:
            # nothing matched return character token
            token = self.text[0]
            self.text = self.text[1:]

        return token


def tokenize(text):
    tk = Tokenizer(text)
    return [token for token in tk]