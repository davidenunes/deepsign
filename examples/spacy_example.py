import spacy
import time

from spacy import parts_of_speech as PoS
from spacy.vocab import Vocab

print("loading spacy")
t0 = time.time()
nlp = spacy.load('en', vocab=Vocab(lex_attr_getters=None))
t1 = time.time()
print("model loaded in {t:.4f} seconds".format(t=t1 - t0))

print("Vocab. Size: ", len(nlp.vocab.strings))

print("PoS Tagging:")
print("tags \n")
for tag in PoS.NAMES:
    print(nlp.vocab.strings[tag])

print("==========================================")
sentence = "Mr Anderson welcome back, we missed you!"
doc = nlp.tokenizer(sentence)

print(doc)
doc = nlp.tagger(doc)
for word in doc:
    print("({w},{t})".format(w=word.orth_, t=word.tag_))

print("Vocab. Size: ", len(nlp.vocab.strings))

doc = nlp(sentence)


def print_subtree(token):
    print("token in head")
    print("(" + token.orth_ + "(" + token.pos_ + ")->" + token.dep_ + "->" + token.head.orth_ + ")")
    if token.dep_ == "pobj":
        print("following preposition path")
        for c in token.head.head.children:
            print("(" + c.orth_ + "(" + c.pos_ + ")->" + c.dep_ + "->" + c.head.orth_ + ")")

    print("tokens in children")
    for c in token.children:
        print("(" + c.orth_ + "(" + c.pos_ + ")->" + c.dep_ + "->" + c.head.orth_ + ")")

    print("siblings")
    for c in token.head.children:
        print("(" + c.orth_ + "(" + c.pos_ + ")->" + c.dep_ + "->" + c.head.orth_ + ")")


#token = doc[3]
print_subtree(doc[2])

# for more examples https://spacy.io/docs#token

token = doc[0]

print(token.dep_ + "(" + token.head.orth_ + "," + token.orth_ + ")")

# print(tokens[0].nbor(1).orth_)
# print(tokens[0].nbor(2).orth_)
# print(tokens[0].nbor(3).orth_)

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
