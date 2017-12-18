import unittest
from deepsign.nlp import is_token
from deepsign.nlp.tokenization import Tokenizer
from deepsign.nlp.tokenization import RE as tokenizer_p

from deepsign.nlp.regex_utils import REMatcher


def test_pattern_sequence(pattern_seq, text):
    """receives a sequence of pattern tuples (name, pattern)
    and tests this sequence against a given text
    """
    matcher = REMatcher()

    for (p_name, p) in pattern_seq:
        if matcher.match(p,text):
            return p_name
    return None

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            "This is a sentence.",
            "This is, without a doubt, a sentence.",
            "Based in Eugene,Ore., PakTech needs a new distributor after Sydney-based Creative Pack Pty. Ltd. went into voluntary administration.",
            "The Iron Age (ca. 1300 -- ca. 300 BC).",
            u"Indo\u00ADnesian ship\u00ADping \u00AD",
            "Gimme a phone, I'embed_size gonna call.",
            "\"John & Mary's dog,\" Jane thought (to herself).\n\"What a #$%!\na- ``I like AT&T''.\"",
            "I said at 4:45pm.",
            "I can't believe they wanna keep 40% of that.\"\n``Whatcha think?''\n\"I don't --- think so...,\"",
            "You `paid' US$170,000?!\nYou should've paid only$16.75.",
            "1. Buy a new Chevrolet (37%-owned in the U.S..) . 15%",
            "I like you ;-) but do you care :(. I'embed_size happy ^_^ but shy (x.x)!",
            "Diamond (``Not even the chair'') lives near Udaipur (84km). {1. A potential Palmer trade:}",
            "No. I like No. 24 and no.47.",
            "You can get a B.S. or a B. A. or a Ph.D (sometimes a Ph. D) from Stanford.",
            "@Harry_Styles didn`t like Mu`ammar al-Qaddafi",
            "Kenneth liked Windows 3.1, Windows 3.x, and Mesa A.B as I remember things.",
            "I like programming in F# more than C#.",
            "NBC Live will be available free through the Yahoo! Chat Web site. E! Entertainment said ``Jeopardy!'' is a game show.",
            "I lived in O\u2019Malley and read OK! Magazine.",
            "I don't give a f**k about your sh*tty life.",
            "First sentence.... Second sentence.",
            "First sentence . . . . Second sentence.",
            "I wasn't really ... well, what I mean...see ... what I'embed_size saying, the thing is ... I didn't mean it.",
            "This is a url test. Here is one: http://google.com.",
            "This is a url test. Here is one: htvp://google.com.",
            "Download from ftp://myname@host.dom/%2Fetc/motd",
            "Download from svn://user@location.edu/path/to/magic/unicorns",
            "Download from svn+ssh://user@location.edu/path/to/magic/unicorns",
            "I dunno, I'ven't done it.",
            "The okay was received by the anti-acquisition front on its foolishness-filled fish market.",
            "We ran the pre-tests through the post-scripted centrifuge.",
            "School-aged parents should be aware of the unique problems that they face.",
            "Ja'net Bri'an O'neill 't'ony",
            "Almost 80 °C or °F "
        ]


        self.gold = [
            ["This", "is", "a", "sentence", "."],
            ["This", "is", ",", "without", "a", "doubt", ",", "a", "sentence","."],
            ["Based", "in", "Eugene", ",", "Ore", ".", ",", "PakTech", "needs", "a", "new", "distributor", "after", "Sydney", "-", "based", "Creative", "Pack", "Pty.", "Ltd.","went", "into", "voluntary", "administration", "."],
            ["The", "Iron", "Age", "(", "ca.", "1300", "--", "ca.", "300", "BC", ")", "."],
            ["Indonesian", "shipping", "-"],
            ["Gimme", "a", "phone", ",", "I", "'embed_size", "gonna", "call", "."],
            ["\"", "John", "&", "Mary", "'s", "dog", ",", "\"", "Jane", "thought", "(", "to", "herself", ")", ".", "\"","What", "a", "#", "$", "%", "!", "a", "-", "``", "I", "like", "AT&T", "''", ".", "\""],
            ["I", "said", "at", "4:45", "pm", "."],
            ["I", "ca", "ngram_size't", "believe", "they", "wanna", "keep", "40", "%", "of", "that", ".", "\"", "``", "Whatcha", "think", "?", "''", "\"", "I", "do", "ngram_size't", "---", "think", "so", "...", ",","\""],
            ["You", "`", "paid", "'", "US", "$", "170,000", "?!", "You", "should", "'ve", "paid", "only", "$", "16.75","."],
            ["1", ".", "Buy", "a", "new", "Chevrolet", "(", "37", "%", "-", "owned", "in", "the", "U.S.", ".", ")", ".","15", "%"],
            ["I", "like", "you", ";-)", "but", "do", "you", "care", ":(", ".","I", "'embed_size", "happy", "^_^", "but", "shy", "(x.x)", "!"],
            ["Diamond", "(", "``", "Not", "even", "the", "chair", "''", ")", "lives", "near", "Udaipur", "(","84","km", ")", ".","{", "1", ".", "A", "potential", "Palmer", "trade", ":", "}"],
            ["No", ".", "I", "like", "No",".", "24", "and", "no", ".", "47", "."],
            ["You", "can", "get", "a", "B.S.", "or", "a", "B.", "A.", "or", "a", "Ph.D", "(", "sometimes", "a", "Ph.","D", ")", "from", "Stanford", "."],
            ["@Harry_Styles", "did", "ngram_size`t", "like", "Mu`ammar", "al","-","Qaddafi"],
            ["Kenneth", "liked", "Windows", "3.1", ",", "Windows", "3.x", ",", "and", "Mesa", "A.B", "as", "I","remember","things", ".", ],
            ["I", "like", "programming", "in", "F#", "more", "than", "C#", "."],
            ["NBC", "Live", "will", "be", "available", "free", "through", "the", "Yahoo", "!", "Chat", "Web", "site", ".","E", "!", "Entertainment", "said", "``", "Jeopardy", "!", "''", "is", "a", "game", "show", "."],
            ["I", "lived", "in", "O\u2019Malley", "and", "read", "OK", "!", "Magazine", "."],
            ["I", "do", "ngram_size't", "give", "a", "f**k", "about", "your", "sh*tty", "life", "."],
            ["First", "sentence", "...", ".", "Second", "sentence", "."],
            ["First", "sentence", ". . .", ".", "Second", "sentence", "."],
            ["I", "was", "ngram_size't", "really", "...", "well", ",", "what", "I", "mean", "...", "see", "...", "what", "I","'embed_size","saying",",", "the", "thing", "is", "...", "I", "did", "ngram_size't", "mean", "it", "."],
            ["This", "is", "a", "url", "test", ".", "Here", "is", "one", ":", "http://google.com", "."],
            ["This", "is", "a", "url", "test", ".", "Here", "is", "one", ":", "htvp://google.com", "."],
            ["Download", "from", "ftp://myname@host.dom/%2Fetc/motd"],
            ["Download", "from", "svn://user@location.edu/path/to/magic/unicorns"],
            ["Download", "from", "svn", "+", "ssh://user@location.edu/path/to/magic/unicorns"],
            ["I", "dunno", ",","I", "'ve", "ngram_size't", "done", "it", "."],
            ["The", "okay", "was", "received", "by", "the", "anti", "-", "acquisition", "front", "on", "its","foolishness","-","filled","fish", "market", "."],
            ["We", "ran", "the", "pre", "-","tests", "through", "the", "post", "-","scripted", "centrifuge", "."],
            ["School", "-", "aged", "parents", "should", "be", "aware", "of", "the", "unique", "problems", "that", "they","face","."],
            ["Ja'net", "Bri'an", "O'neill", "'", "t'ony"],
            ["Almost", "80", "°C", "or", "°F"]
        ]

    def test_sentence(self):
        sentence = "benzyldimethyldodecylammonium chloride , benzyldimethyltetradecylammonium chloride , and benzyldimethylhexadecylammonium chloride , commonly know as benzalkonium chloride ."
        exptected = ["benzyldimethyldodecylammonium", "chloride", ",", "benzyldimethyltetradecylammonium chloride", ",", "and", "benzyldimethylhexadecylammonium", "chloride", ",", "commonly", "know", "as", "benzalkonium", "chloride", "."]

        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(sentence)



    def test_tokenizer(self):
        """ Tests a single sentence:
        Useful to debug the tokenizer during development.
        Just use one of the sentences above along with its gold standard
        or enter a new one bellow
        """

        tokenizer = Tokenizer()

        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            gold = self.gold[i]

            tokens = [t for t in tokenizer.tokenize(sentence) if not is_token.is_space(t)]
            self.assertSequenceEqual(tokens,gold)

    def test_on_url(self):
        tokenizer = Tokenizer()

        pattern_name_seq = [
            "SPACES",
            "URL",
            "EMAIL",
            "ABBREVIATION",
            "NUMERIC",
            "CONTRACTION",
            "CONTRACTION_W1",
            "CONTRACTION_W2",
            "CONTRACTION_W3",
            "CONTRACTION_W4",
            "CONTRACTION_WE",
            "CONTRACTION_WO",
            "CENSORED_WORD",
            "SOFT_HYPHEN",
            "PROGRAMMING_LANGUAGES",
            "WORD",
            "HASHTAG",
            "USER_HANDLE",
            "EMOTICON",
            "DEGREES",
            "PUNCT"
        ]

        pattern_seq = [(name,tokenizer_p[name]) for name in pattern_name_seq]

        url1 = "www.newstatesman.co.uk/upstarts/upstarts2002shortlist.htm"
        url2 = "www.south-west.org.uk"
        url3 = "www.toolpost.co.uk"
        url4 = "www.fairplay.org.uk"

        self.assertEqual("URL",test_pattern_sequence(pattern_seq,url1))

        self.assertEqual("URL",test_pattern_sequence(pattern_seq,url2))

        self.assertEqual("URL",test_pattern_sequence(pattern_seq,url3))

        self.assertEqual("URL",test_pattern_sequence(pattern_seq,url4))





