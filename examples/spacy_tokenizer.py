from spacy.en import English


print("Loading English Model...")
nlp =  English(entity=False,tagger=False,parser=False)
print("Done!")

print("Vocab. Size: ",len(nlp.vocab.strings))
print("hello" in nlp.vocab.strings)

#loads the entire nlp pipeline with parser, named-entity recognition, pos tagger
tokens = nlp(u'Mr Anderson, welcome back, we missed you.')
tokens = [token for token in tokens]
print(tokens)

tokens = nlp.tokenizer('Mr Anderson, welcome back, we missed you.')
print(type(tokens))
tokens = [token for token in tokens]
print(tokens)

print(type(tokens[0]))
print(tokens[0].orth_)