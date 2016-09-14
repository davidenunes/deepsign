import unittest
from deepsign.nlp import patterns
from deepsign.nlp import token
from segtok.tokenizer import web_tokenizer
import re

class TestPatterns(unittest.TestCase):
    def test_url(self):

        url1 = "http://stuff.com?stuff=cenas"
        url2 = "http://davikjhasds.xyz"
        url3 = "mailto:theman@theboss.com"
        email = "esbanhsdoias@mail.com"

        assert(token.is_url(url1))
        assert(token.is_url(url2))
        assert(token.is_url(url3))
        assert(token.is_url(email))

        assert(not token.is_email(url1))
        assert(not token.is_email(url2))
        assert(not token.is_email(url3))
        assert(token.is_email(email))

        sentence = "stuff and other stuff ahaha (" + url1 + ") final stuff"

        url1_re = patterns.URL
        email_re = patterns.EMAIL

        print(url1_re.split(sentence))
        print(web_tokenizer(sentence))

