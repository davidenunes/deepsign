#!/usr/bin/env python
"""
RexEx Tokenizer
"""

from deepsign.nlp import token

def split_contractions(tokens):
    """
    A function to split apostrophe contractions at the end of alphanumeric (and hyphenated) tokens.
    Takes the output of any of the tokenizer functions and produces and updated list.
    :param tokens: a list of tokens
    :returns: an updated list if a split was made or the original list otherwise
    """
    idx = -1

    for token in list(tokens):
        idx += 1

        if IS_CONTRACTION.match(token) is not None:
            length = len(token)

            if length > 1:
                for pos in range(length - 1, -1, -1):
                    if token[pos] in APOSTROPHES:
                        if 2 < length and pos + 2 == length and token[-1] == 't' and token[pos - 1] == 'n':
                            pos -= 1

                        tokens.insert(idx, token[:pos])
                        idx += 1
                        tokens[idx] = token[pos:]

    return tokens
