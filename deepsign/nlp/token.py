import unicodedata
import re
from deepsign.nlp import patterns



def is_punct(token):
    for c in token:
        if not unicodedata.category(c).startswith('P'):
            return False

    return True


def is_bracket(token):
    # Penn Treebank bracket tokens
    ptb_brackets = ('-LRB-', '-RRB-', '-RSB-', '-RSB-', '-LCB-', '-RCB-')
    brackets = ('(', ')', '[', ']', '{', '}', '<', '>',) + ptb_brackets

    return token in brackets


def is_quote(token):
    quotes = ('"', "'", '`', '«', '»', '‘', '’', '‚', '‛', '“', '”', '„', '‟', '‹', '›', '❮', '❯', "''", '``')
    return token in quotes


def is_left_punct(token):
    left_punct = ('(', '[', '{', '<', '"', "'", '«', '‘', '‚', '‛', '“', '„', '‟', '‹', '❮', '``')
    return token in left_punct


def is_right_punct(token):
    right_punct = (')', ']', '}', '>', '"', "'", '»', '’', '”', '›', '❯', "''")
    return token in right_punct

_url_re = patterns.RE_URL


def is_url(token):
    return _url_re.match(token) is not None


_email_re = re.compile(patterns.RE_EMAIL)

def is_email(token):
    return _email_re.match(token) is not None
