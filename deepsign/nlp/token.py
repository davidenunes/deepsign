import unicodedata
import re
from deepsign.nlp import patterns


def is_punct(text):
    for c in text:
        if not unicodedata.category(c).startswith('P'):
            return False

    return True


def is_parens_bracket(text):
    return re.fullmatch(patterns.PARENS_BRACKET, text) is not None


def is_quote(text):
    return re.fullmatch(patterns.QUOTES, text) is not None


def is_url(text):
    return patterns.RE_URL.fullmatch(text) is not None


def is_email(text):
    return patterns.RE_EMAIL.fullmatch(text) is not None
