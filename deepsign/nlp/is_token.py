"""
I could do full match, but in some cases if something doesn't match there are patterns that take a long time to decide
weather or not something matches (e.g. a bad url). To guarantee that this doesn't happen we trust the tokens are
segmented properly.

For example something like "(this" will return true on is_parens_bracket because it starts with a parenthesis
"""

import re
from deepsign.nlp import patterns, stoplist


def is_numeric(text):
    return re.match(patterns.NUMERIC, text) is not None


def is_word(text):
    return re.match(patterns.WORD, text) is not None


def is_space(text):
    return re.match(patterns.SPACES, text) is not None


def is_punct(text):
    return re.match(patterns.PUNCT, text) is not None


def is_parens_bracket(text):
    return re.match(patterns.PARENS_BRACKET, text) is not None


def is_quote(text):
    return re.match(patterns.QUOTE, text) is not None


def is_url(text):
    # TODO have a compile or match function for each pattern
    # a pattern would have to be an object with a custom match function
    # with the required flags so that people don't need to inspect code
    return re.match(patterns.URL, text, re.VERBOSE | re.UNICODE) is not None


def is_email(text):
    return re.match(patterns.EMAIL, text, re.VERBOSE | re.UNICODE) is not None


def is_currency(text):
    return re.match(patterns.CURRENCY, text) is not None


def is_stopword(text):
    return text.lower() in stoplist.ENGLISH


def is_copyright(text):
    return re.match(patterns.COPYRIGHT, text) is not None

# TODO add patterns for time
# Days of the Week, Months, Years, Today, Yesterday, Tomorrow
# Abbrev of Months
# Abbrev of epochs in the 60s
# Decades sixties
