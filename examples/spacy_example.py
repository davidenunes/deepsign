from spacy.en import English
import spacy.parts_of_speech as POS


print("Loading English Model...")
nlp =  English()
print("Done!")


print("Available Parts-Of-Speech:")
print(POS.NAMES)

VERB = nlp.vocab.strings['VERB']
print(VERB)

#loads the entire nlp pipeline with parser, named-entity recognition, pos tagger
tokens = nlp(u'Mr Anderson, welcome back, we missed you.')
print(tokens)


def print_subtree(token):
    print("token in head")
    print("(" + token.orth_ + "(" + token.pos_ + ")--->" + token.dep_ + "--->" + token.head.orth_ + ")")
    if token.dep_ =="pobj":
        print("following preposition path")
        for c in token.head.head.children:
            print("(" + c.orth_ + "(" + c.pos_ + ")--->" + c.dep_ + "--->" + c.head.orth_ + ")")

    print("tokens in children")
    for c in token.children:
        print("(" + c.orth_ + "(" + c.pos_ + ")--->" + c.dep_ + "--->" + c.head.orth_ + ")")

    print("siblings")
    for c in token.head.children:
        print("(" + c.orth_ + "(" + c.pos_ + ")--->" + c.dep_ + "--->" + c.head.orth_ + ")")


token = tokens[3]
print_subtree(token)


# for more examples https://spacy.io/docs#token

token = tokens[0]

print(token.dep_+"("+token.head.orth_+","+token.orth_+")")

#print(tokens[0].nbor(1).orth_)
#print(tokens[0].nbor(2).orth_)
#print(tokens[0].nbor(3).orth_)

# testing determinant dependencies
tokens = nlp(u'The cat eats the mouse and carrots.')
print(tokens)

token = tokens[1]
print_subtree(token)

# testing determinant dependencies
tokens = nlp(u'The cat is black.')
print(tokens)

token = tokens[1]
print_subtree(token)

tokens = nlp(u'The mouse is eaten by the cat.')
print(tokens)

token = tokens[6]
print_subtree(token)


tokens = nlp(u'The mouse has legs')
print(tokens)

token = tokens[1]
print_subtree(token)

tokens = nlp(u'The mouse is walking')
print(tokens)

token = tokens[1]
print_subtree(token)


