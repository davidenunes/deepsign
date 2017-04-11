from deepsign.nlp import is_token as itk
from deepsign.utils.listx import match_replace
import spacy

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
    if (itk.is_punct(token) or itk.is_space(token) or itk.is_copyright(token) or token in _wacky_tokens):
        return True
    return False


class WaCKyPipe:
    def __init__(self, datagen):
        self.datagen = datagen
        print("Loading Spacy EN Model...")
        self.reaload()
        print("done")


    def __iter__(self):
        return self

    def __next__(self):
        sentence = next(self.token_gen)
        tokens = [token.orth_ for token in sentence]
        tokens = [match_replace(token,replacements) for token in tokens if not invalid_token(token)]
        tokens = [token.lower() for token in tokens]

        #sentence = next(self.datagen)
        #tokens = self.tokenizer.tokenize(sentence)

        return tokens


    def reaload(self):
        self.nlp = spacy.load("en", entity=False, parser=False, vectors=False)
        self.token_gen = self.nlp.pipe(self.datagen, batch_size=10000, n_threads=4)
        #self.token_gen = (self.nlp(sentence) for sentence in self.datagen)


