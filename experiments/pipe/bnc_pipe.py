from deepsign.nlp import is_token as itk
import spacy


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
    def __init__(self, datagen,lemmas=False):
        self.lemmas = lemmas

        print("Loading Spacy EN Model...")
        self.reaload(datagen, load_model=True)

        print("done")

        #self.tk = Tokenizer()

    def __iter__(self):
        return self

    def __next__(self):
        sentence = next(self.token_gen)
        if self.lemmas:
            tokens = [token.lemma_ for token in sentence]
        else:
            tokens = [token.orth_ for token in sentence]
        tokens = [token.lower() for token in tokens if not invalid_token(token)]


        return tokens

    def reaload(self,datagen,load_model=False,):
        # I just need the tagger for lemmas

        if load_model:
            self.nlp = spacy.load("en")
        #self.nlp = English(entity=False, parser=False)
        self.token_gen = self.nlp.pipe(datagen, batch_size=4000, n_threads=3)
        #self.token_gen = (self.nlp(sentence) for sentence in self.datagen)


