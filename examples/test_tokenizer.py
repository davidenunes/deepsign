from deepsign.nlp.tokenization import Tokenizer

sentence = "hello there mr anderson!"

tokenizer = Tokenizer()

tokens = tokenizer.tokenize(sentence)
print(tokens)


