from spacy.en import English

print("Loading English Model...")
nlp =  English()
print("Done!")

#loads the entire nlp pipeline with parser, named-entity recognition, pos tagger
tokens = nlp(u'Mr Anderson, welcome back, we missed you.')
print(tokens)

#for t in tokens:
 #   print("("+t.dep_ + " "+t.orth_+")")

token = tokens[0]

while token.head is not token:
    print("("+token.orth_+"--->"+token.dep_+"--->"+token.head.orth_+")")
    token = token.head


# for more examples https://spacy.io/docs#token

token = tokens[0]

print(token.dep_+"("+token.head.orth_+","+token.orth_+")")

#print(tokens[0].nbor(1).orth_)
#print(tokens[0].nbor(2).orth_)
#print(tokens[0].nbor(3).orth_)
