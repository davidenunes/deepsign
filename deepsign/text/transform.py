import deepsign.text.tokenizer as tk
import re


def to_lower(seq=[]):
        return [s.lower() for s in seq]


def rm_punctuation(seq=[]):
        CLITIC_REGEX = tk.clitic
        WORD_REGEX = "\w+"
        VALID_REGEX = re.compile(tk.regex_or(CLITIC_REGEX,WORD_REGEX))

        def filterfn(word):
            if VALID_REGEX.search(word):
                return True
            else:
                return False

        return list(filter(filterfn,seq))

