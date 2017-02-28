from deepsign.nlp import is_token as itk
from deepsign.utils.listx import match_replace

from spacy.en import English
from deepsign.nlp.tokenization import Tokenizer




# token replacement rules
replacements = (
    (itk.is_url, "T_URL"),
    (itk.is_email, "T_EMAIL"),
    (itk.is_currency, "T_CURRENCY")
)


def invalid_token(token):
    if (itk.is_punct(token) or itk.is_space(token)):
        return True
    return False


class BNCPipe:
    def __init__(self, datagen):
        self.datagen = datagen

        print("Loading Spacy EN Model...")
        self.reaload()
        print("done")

        #self.tk = Tokenizer()

    def __iter__(self):
        return self

    def __next__(self):
        sentence = next(self.token_gen)
        #tokens = self.tokenizer.tokenize(sentence)
        #tokens = sentence.split()
        tokens = [token.orth_ for token in sentence]
        #tokens = self.tk.tokenize(sentence)
        #tokens = [match_replace(token,replacements) for token in tokens if not invalid_token(token)]
        tokens = [token.lower() for token in tokens if not invalid_token(token)]


        return tokens

    def reaload(self):
        self.nlp = English(entity=False, tagger=False, parser=False)
        self.token_gen = self.nlp.pipe(self.datagen, batch_size=10000, n_threads=4)
        #self.token_gen = (self.nlp(sentence) for sentence in self.datagen)


