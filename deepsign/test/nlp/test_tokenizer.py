import unittest
from deepsign.nlp import patterns
from deepsign.nlp import token
from deepsign.nlp.tokenizer import tokenize
from segtok.tokenizer import web_tokenizer
import re


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            "This is a sentence.",
            "This is, without a doubt, a sentence.",
            "Based in Eugene,Ore., PakTech needs a new distributor after Sydney-based Creative Pack Pty. Ltd. went into voluntary administration.",
            "The Iron Age (ca. 1300 – ca. 300 BC).",
            "Indo\u00ADnesian ship\u00ADping \u00AD",
            "Gimme a phone, I'm gonna call.",
            "\"John & Mary's dog,\" Jane thought (to herself).\n\"What a #$%!\na- ``I like AT&T''.\"",
            "I said at 4:45pm.",
            "I can't believe they wanna keep 40% of that.\"\n``Whatcha think?''\n\"I don't --- think so...,\"",
            "You `paid' US$170,000?!\nYou should've paid only$16.75.",
            "1. Buy a new Chevrolet (37%-owned in the U.S..) . 15%",
            "I like you ;-) but do you care :(. I'm happy ^_^ but shy (x.x)!",
            "Diamond (``Not even the chair'') lives near Udaipur (84km). {1. A potential Palmer trade:}",
            "No. I like No. 24 and no.47.",
            "You can get a B.S. or a B. A. or a Ph.D (sometimes a Ph. D) from Stanford.",
            "@Harry_Styles didn`t like Mu`ammar al-Qaddafi",
            "Kenneth liked Windows 3.1, Windows 3.x, and Mesa A.B as I remember things.",
            "I like programming in F# more than C#.",
            "NBC Live will be available free through the Yahoo! Chat Web site. E! Entertainment said ``Jeopardy!'' is a game show.",
            "I lived in O\u2019Malley and read OK! Magazine.",
            "I lived in O\u0092Malley and read OK! Magazine.",
            "I don't give a f**k about your sh*tty life.",
            "First sentence.... Second sentence.",
            "First sentence . . . . Second sentence.",
            "I wasn’t really ... well, what I mean...see . . . what I'm saying, the thing is . . . I didn’t mean it.",
            "This is a url test. Here is one: http://google.com.",
            "This is a url test. Here is one: htvp://google.com.",
            "Download from ftp://myname@host.dom/%2Fetc/motd",
            "Download from svn://user@location.edu/path/to/magic/unicorns",
            "Download from svn+ssh://user@location.edu/path/to/magic/unicorns",
            "We traveled from No. Korea to So. Calif. yesterday.",
            "I dunno.",
            "The o-kay was received by the anti-acquisition front on its foolishness-filled fish market.",
            "We ran the pre-tests through the post-scripted centrifuge.",
            "School-aged parents should be aware of the unique problems that they face."]

        self.gold = [
            ["This", "is", "a", "sentence", "."],
            ["This", "is", ",", "without", "a", "doubt", ",", "a", "sentence","."],
            ["Based", "in", "Eugene", ",", "Ore.", ",", "PakTech", "needs", "a", "new", "distributor", "after", "Sydney-based", "Creative", "Pack", "Pty.", "Ltd.","went", "into", "voluntary", "administration", "."],
            ["The", "Iron", "Age", "(", "ca.", "1300", "--", "ca.", "300", "BC", ")", "."],
            ["Indonesian", "shipping", "-"],
            ["Gim", "me", "a", "phone", ",", "I", "'m", "gon", "na", "call", "."],
            ["``", "John", "&", "Mary", "'s", "dog", ",", "''", "Jane", "thought", "(", "to", "herself", ")", ".", "``","What", "a", "#", "$", "%", "!", "a", "-", "``", "I", "like", "AT&T", "''", ".", "''"],
            ["I", "said", "at", "4:45", "pm", "."],
            ["I", "ca", "n't", "believe", "they", "wan", "na", "keep", "40", "%", "of", "that", ".", "''", "``", "Whatcha", "think", "?", "''", "``", "I", "do", "n't", "--", "think", "so", "...", ",","''"],
            ["You", "`", "paid", "'", "US$", "170,000", "?!", "You", "should", "'ve", "paid", "only", "$", "16.75","."],
            ["1", ".", "Buy", "a", "new", "Chevrolet", "(", "37", "%", "-", "owned", "in", "the", "U.S.", ".", ")", ".","15", "%"],
            ["I", "like", "you", ";-)", "but", "do", "you", "care", ":(", ".","I", "'m", "happy", "^_^", "but", "shy", "(x.x)", "!"],
            ["Diamond", "(", "``", "Not", "even", "the", "chair", "''", ")", "lives", "near", "Udaipur", "(","84km", ")", ".","{", "1", ".", "A", "potential", "Palmer", "trade", ":", "}"],
            ["No", ".", "I", "like", "No.", "24", "and", "no.", "47", "."],
            ["You", "can", "get", "a", "B.S.", "or", "a", "B.", "A.", "or", "a", "Ph.D", "(", "sometimes", "a", "Ph.","D", ")", "from", "Stanford", "."],
            ["@Harry_Styles", "did", "n`t", "like", "Mu`ammar", "al-Qaddafi"],
            ["Kenneth", "liked", "Windows", "3.1", ",", "Windows", "3.x", ",", "and", "Mesa", "A.B", "as", "I","remember","things", ".", ],
            ["I", "like", "programming", "in", "F#", "more", "than", "C#", "."],
            ["NBC", "Live", "will", "be", "available", "free", "through", "the", "Yahoo!", "Chat", "Web", "site", ".","E!", "Entertainment", "said", "``", "Jeopardy!", "''", "is", "a", "game", "show", "."],
            ["I", "lived", "in", "O'Malley", "and", "read", "OK!", "Magazine", "."],
            ["I", "lived", "in", "O'Malley", "and", "read", "OK!", "Magazine", "."],
            ["I", "do", "n't", "give", "a", "f**k", "about", "your", "sh*tty", "life", "."],
            ["First", "sentence", "...", ".", "Second", "sentence", "."],
            ["First", "sentence", "...", ".", "Second", "sentence", "."],
            ["I", "was", "n't", "really", "...", "well", ",", "what", "I", "mean", "...", "see", "...", "what", "I","'m","saying",",", "the", "thing", "is", "...", "I", "did", "n't", "mean", "it", "."],
            ["This", "is", "a", "url", "test", ".", "Here", "is", "one", ":", "http://google.com", "."],
            ["This", "is", "a", "url", "test", ".", "Here", "is", "one", ":", "htvp", ":", "/", "/", "google.com", "."],
            ["Download", "from", "ftp://myname@host.dom/%2Fetc/motd"],
            ["Download", "from", "svn://user@location.edu/path/to/magic/unicorns"],
            ["Download", "from", "svn+ssh://user@location.edu/path/to/magic/unicorns"],
            ["We", "traveled", "from", "No.", "Korea", "to", "So.", "Calif.", "yesterday", "."],
            ["I", "du", "n", "no", "."],
            ["The", "o-kay", "was", "received", "by", "the", "anti-acquisition", "front", "on", "its","foolishness-filled","fish", "market", "."],
            ["We", "ran", "the", "pre-tests", "through", "the", "post-scripted", "centrifuge", "."],
            ["School-aged", "parents", "should", "be", "aware", "of", "the", "unique", "problems", "that", "they","face","."]
        ]

    def test_one_sentence(self):
        """ Tests a single sentence:
        Useful to debug the tokenizer during development.
        Just use one of the sentences above along with its gold standard
        or enter a new one bellow
        """
        s_index = 1

        sentence = self.sentences[s_index]
        gold = self.gold[s_index]

        print(sentence)
        print(gold)

        tokens = tokenize(sentence)
        print(tokens)

        self.assertSequenceEqual(tokens,gold)


