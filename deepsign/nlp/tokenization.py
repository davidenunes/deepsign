#!/usr/bin/env python
"""
RexEx Tokenizer

NOTE: A space or a sequence of spaces is returned as a token
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
    'CONTRACTION_W4': re.compile(pm.CONTRACTION_WORD_4, re.UNICODE),
    'CONTRACTION_WE': re.compile(pm.CONTRACTION_WORD_EXTRA, re.UNICODE),
    'CONTRACTION_WO': re.compile(pm.CONTRACTION_WORD_OTHER, re.UNICODE),
    'URL': re.compile(pm.URL, re.UNICODE | re.VERBOSE),
    'EMAIL': re.compile(pm.EMAIL, re.UNICODE | re.VERBOSE),
    'CENSORED_WORD': re.compile(pm.WORD_CENSORED, re.UNICODE),
    'ABBREVIATION': re.compile(pm.ABBREV, re.UNICODE),
    'SOFT_HYPHEN': re.compile(pm.SOFT_HYPHEN, re.UNICODE),
    'PROGRAMMING_LANGUAGES': re.compile(pm.PROGRAMMING_LANGUAGES, re.UNICODE),
    'WORD': re.compile(pm.WORD, re.UNICODE),
    'NUMERIC': re.compile(pm.NUMERIC, re.UNICODE),
    'HASHTAG': re.compile(pm.HASHTAG, re.UNICODE),
    'USER_HANDLE': re.compile(pm.TWITTER_HANDLE, re.UNICODE),
    'EMOTICON': re.compile(pm.EMOTICON, re.UNICODE),
    'DEGREES': re.compile(pm.DEGREES, re.UNICODE),
    'PUNCT': re.compile(pm.PUNCT, re.UNICODE)
}

class Tokenization:
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
        token = None
        match_group = -1
        rm_soft_hyphen = False

        # 0 is the whole match
        # 1 the first group (if it exists)
        if matcher.match(RE['SPACES'], text):
            match_group = 0
        elif matcher.match(RE['URL'], text):
            # match url and uri but not e-mail
            match_group = 0
        elif matcher.match(RE['EMAIL'], text):
            match_group = 0
        elif matcher.match(RE['ABBREVIATION'], text):
            # etc. Ph.D p.embed_size. A.M.
            match_group = 0
        elif matcher.match(RE['NUMERIC'], text):
            match_group = 0
            rm_soft_hyphen = True
        elif matcher.match(RE['CONTRACTION'], text):
            # (do)(ngram_size't) -> return (do) ->  skip to (ngram_size't)
            match_group = 0
            rm_soft_hyphen = True
        elif matcher.match(RE['CONTRACTION_W1'], text):
            # I'embed_size
            match_group = 1
        elif matcher.match(RE['CONTRACTION_W2'], text):
            # y'all y'know
            match_group = 1
        elif matcher.match(RE['CONTRACTION_W3'], text):
            # He's He'd woudn't've Mary's
            match_group = 1
        elif matcher.match(RE['CONTRACTION_W4'], text):
            # 'tis 'twas
            match_group = 1
        elif matcher.match(RE['CONTRACTION_WE'], text):
            # words with apostrophe from a list (w) -> (w)
            match_group = 0
        elif matcher.match(RE['CONTRACTION_WO'], text):
            # any other words with apostrophe (w) -> (w)
            match_group = 0
        elif matcher.match(RE['CENSORED_WORD'], text):
            # match f***k
            match_group = 0
            rm_soft_hyphen = True
        elif matcher.match(RE['SOFT_HYPHEN'], text):
            # substitute the match
            match_group = 0
            token = "-"
        elif matcher.match(RE['PROGRAMMING_LANGUAGES'], text):
            match_group = 0
        elif matcher.match(RE['WORD'], text):
            match_group = 0
            rm_soft_hyphen = True
        elif matcher.match(RE['HASHTAG'], text):
            match_group = 0
        elif matcher.match(RE['USER_HANDLE'], text):
            match_group = 0
        elif matcher.match(RE['EMOTICON'], text):
            match_group = 0
        elif matcher.match(RE['DEGREES'], text):
            match_group = 0
        elif matcher.match(RE['PUNCT'], text):
            match_group = 0

        # for hardcoded replaces
        if token is None:
            if match_group >= 0:
                self.text = matcher.skip(match_group)
                token = matcher.matched.group(match_group)

                if rm_soft_hyphen:
                    token = re.sub("\u00AD", "", token)
            else:
                # nothing matched return character token
                token = self.text[0]
                self.text = self.text[1:]
        else:
            self.text = matcher.skip(match_group)

        return token


class Tokenizer:
    def tokenize(self, text):
        tk = Tokenization(text)
        return [token for token in tk]
