from deepsign.nlp import is_token as itk
from deepsign.utils.listx import match_replace
from deepsign.utils.views import chunk_it

# token replacement rules
replacements = (
    (itk.is_url, "T_URL"),
    (itk.is_email, "T_EMAIL"),
    (itk.is_currency, "T_CURRENCY")
)


# invalid token functions
# TODO check what other tokens appear in wacky
# TODO the custom tokens should be detected directly tokenizer as custom rules
# these appear as @card@ but are tokenised as @card @...
_wacky_tokens = ("@card", "@ord")


def invalid_token(token):
    if (itk.is_punct(token)
        or itk.is_space(token)
        or itk.is_copyright(token)
            or token in _wacky_tokens):
        return True
    return False


class WaCKyPipe:
    def __init__(self,datagen,tokenizer,filter_stop=False):
        self.datagen = datagen
        self.tokenizer = tokenizer
        self.filter_stop = filter_stop

    def __iter__(self):
        return self

    def __next__(self):
        sentence = next(self.datagen)
        tokens = self.tokenizer.tokenize(sentence)
        tokens = [match_replace(token,replacements) for token in tokens if not invalid_token(token)]
        tokens = [token.lower() for token in tokens]
        if self.filter_stop:
            tokens = [token for token in tokens if not itk.is_stopword(token)]

        return tokens


