import deepsign.text.tokenizer as tk
import re


class Transform:
    def apply(self,seq=[]):
        return seq


class TransformPipeline:
    def __init__(self):
        self.pipeline = []

    def append(self, transform):
        if not isinstance(transform,Transform):
            raise Exception("cannot only append transform instances")
        self.pipeline.append(transform)

    def apply(self,seq):
        res = seq
        for tf in self.pipeline:
            res = tf.apply(res)
        return res


class CaseTransform(Transform):
    def apply(self,seq=[]):
        return [s.lower() for s in seq]


class CleanPunctuation(Transform):

    def apply(self,seq=[]):
        CLITIC_REGEX = tk.clitic
        WORD_REGEX = "\w+"
        VALID_REGEX = re.compile(tk.regex_or(CLITIC_REGEX,WORD_REGEX))

        def filterfn(word):
            if VALID_REGEX.search(word):
                return True
            else:
                return False

        return list(filter(filterfn,seq))

