import unittest
from deepsign.nlp import patterns
from deepsign.nlp import token
from segtok.tokenizer import web_tokenizer
import re


def _matches(pattern, target_str):
    found = pattern.findall(target_str)
    return len(found) == 1 and found[0] == target_str

class TestPatterns(unittest.TestCase):



    def test_url_uri_email(self):
        # Examples to be Accepted

        # URL
        url_simple = "domain.tld"
        url_simple_sub = "sub.domain.tld"
        reject_url_simple_sub = "sub..domain.tld"
        url_simple_port = "domain.com:8080"
        url_simple_path1 = "domain.com/file.html"
        url_simple_path2 = "domain.com/path-path"
        url_simple_path3 = "domain.com/path+path2+path3"

        assert (patterns.URL.match(url_simple) )
        assert (patterns.URL.match(url_simple_sub))
        assert (not patterns.URL.match(reject_url_simple_sub))
        assert (patterns.URL.match(url_simple_port))
        assert (patterns.URL.match(url_simple_path1))
        assert (patterns.URL.match(url_simple_path2))
        assert (patterns.URL.match(url_simple_path3))



        url_scheme_simple = "https://domain.tld"
        url_scheme_sub = "https://sub.domain.tld"
        url_path_asterisk = "http://domain.tld/path/*/http://google.com"
        url_path_bang = "http://domain.tld/path/!/http://google.com"
        url_parenthesis_1 = "http://example.com/file%2811%281.html)"
        url_parenthesis_2 = "http://example.com/@user/file(1).html"
        url_parenthesis_3 = "http://sub.domain.domain/p1/r2.4.1/file.html#segment(string)"

        assert (patterns.URL.match(url_scheme_simple))
        assert (patterns.URL.match(url_scheme_sub))
        assert (patterns.URL.match(url_path_asterisk))
        assert (patterns.URL.match(url_parenthesis_1))
        #print(patterns.URL.findall(url_parenthesis_1))
        #print(patterns.URL.findall(url_parenthesis_2))

        print(patterns.URL.findall(url_path_asterisk))
        print(patterns.URL.findall(url_path_bang))
        print(patterns.URL.findall(url_parenthesis_1))
        print(patterns.URL.findall(url_parenthesis_2))
        print(patterns.URL.findall(url_parenthesis_3))

        url_fragment = "http://domain.tld/path/#hello"
        url_handle = "https://domain.tld/@username"
        url_tilde = "http://domain.tld/~username"

        url_path_query_1 = "domain.com/path?key=value&key2=value2"
        url_path_query_2 = "domain.com/path?key=value&key2=value_2"
        url_path_query_3 = "domain.com/path?key=value1+value_2"
        url_path_query_4 = "domain.com/path/!/value1&value2"
        url_path_query_5 = "domain.com/path$?key=value1"
        url_path_query_6 = "domain.com/path?key=value1;value_2"
        url_path_query_7 = "domain.com/path?key=value1%20value_2"

        uri_user = "user:info@host.name.tld"
        uri_mail = "mailto:user@mail.tld"
        uri_full = "(abc://username@example.com:123/path/data?key=value&key2=value2#fragid1)"

        print(patterns.URL.findall(uri_full))


        url_ssh_1 = "user1@domain.com:user2/file.ext"
        url_ssh_2 = "ssh://username@hostname:/path"
        url_ssh_3 = "ssh://username@hostname:1010/path"

        # E-MAIL
        email_simple = "user101-a_b@mail.com"
        email_sub = "yser@one.two.tld"
        email_dotuser = "user.101@mail.tld"

        # sentence = "stuff and other stuff ahaha (" + url1 + ") final stuff"

        # url1_re = patterns.URL
        email_re = patterns.EMAIL

        # print(url1_re.split(sentence))
        # print(web_tokenizer(sentence))
