import unittest
from deepsign.nlp import patterns as pm
from deepsign.nlp import token
from segtok.tokenizer import web_tokenizer
import re


def _matches(pattern, target_str):
    found = pattern.findall(target_str)
    return len(found) == 1 and found[0] == target_str


def print_matches(pattern_dict, txt):
    print("-------------------")
    print("matching: "+txt)
    for k in pattern_dict:
        p = pattern_dict[k]
        m = re.match(p, txt)
        if m is not None:
            print(k + ": " + p)
            print(m.group(0))
    print("-------------------")


class TestPatterns(unittest.TestCase):
    def test_url(self):
        # Examples to be Accepted

        # URL
        url_simple = "domain.tld"
        url_simple_sub = "sub.domain.tld"
        reject_url_simple_sub = "sub..domain.tld"
        url_simple_port = "domain-port.com:8080"
        url_protocol_port = "http://domain-port.com:8080"
        url_simple_path1 = "domain.com/file.html"
        url_simple_protocol_path1 = "https://domain.com/file.html"
        url_simple_path2 = "domain.com/path-path"
        url_simple_protocol_path2 = "http://domain.com/path-path"
        url_simple_path3 = "domain.com/path+path2+path3"
        url_simple_protocol_path3 = "http://domain.com/path+path2+path3"

        self.assertTrue(_matches(pm.RE_URL, url_simple))
        self.assertTrue(_matches(pm.RE_URL, url_simple_sub))
        self.assertFalse(_matches(pm.RE_URL, reject_url_simple_sub))
        self.assertTrue(_matches(pm.RE_URL, url_simple_port))
        self.assertTrue(_matches(pm.RE_URL, url_protocol_port))
        self.assertTrue(_matches(pm.RE_URL, url_simple_path1))
        self.assertTrue(_matches(pm.RE_URL, url_simple_protocol_path1))
        self.assertTrue(_matches(pm.RE_URL, url_simple_path2))
        self.assertTrue(_matches(pm.RE_URL, url_simple_protocol_path2))
        self.assertTrue(_matches(pm.RE_URL, url_simple_path3))
        self.assertTrue(_matches(pm.RE_URL, url_simple_protocol_path3))

        url_scheme_simple = "https://domain.tld"
        url_scheme_sub = "https://sub.domain.tld"
        url_path_asterisk = "http://domain.tld/path/*/http://google.com"
        url_path_bang = "http://domain.tld/path/!/http://google.com"
        url_parenthesis_invalid_1 = "http://example.com/file%2811%281.html)"
        url_parenthesis_2 = "http://example.com/@user/file(1).html"
        url_parenthesis_3 = "http://sub.domain.domain/p1/r2.4.1/file.html#segment(string)"

        self.assertTrue(_matches(pm.RE_URL, url_scheme_simple))
        self.assertTrue(_matches(pm.RE_URL, url_scheme_sub))
        self.assertTrue(_matches(pm.RE_URL, url_path_asterisk))
        self.assertTrue(_matches(pm.RE_URL, url_path_bang))
        self.assertFalse(_matches(pm.RE_URL, url_parenthesis_invalid_1))
        self.assertTrue(_matches(pm.RE_URL, url_parenthesis_2))
        self.assertTrue(_matches(pm.RE_URL, url_parenthesis_3))

        url_fragment = "http://domain.tld/path/#hello"
        url_handle = "https://domain.tld/@username"
        url_tilde = "http://domain.tld/~username"

        self.assertTrue(_matches(pm.RE_URL, url_fragment))
        self.assertTrue(_matches(pm.RE_URL, url_handle))
        self.assertTrue(_matches(pm.RE_URL, url_tilde))

        url_path_query_1 = "domain.com/path?key=value&key2=value2"
        url_path_query_2 = "domain.com/path?key=value&key2=value_2"
        url_path_query_3 = "domain.com/path?key=value1+value_2"
        url_path_query_4 = "domain.com/path/!/value1&value2"
        url_path_query_5 = "domain.com/path$?key=value1"
        url_path_query_6 = "domain.com/path?key=value1;value_2"
        url_path_query_7 = "domain.com/path?key=value1%20value_2"

        self.assertTrue(_matches(pm.RE_URL, url_path_query_1))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_2))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_3))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_4))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_5))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_6))
        self.assertTrue(_matches(pm.RE_URL, url_path_query_7))

        uri_user = "http://user:info@host.name.tld"
        uri_user_2 = "user:info@host.name.tld"
        uri_mail = "mailto:user@mail.tld"
        uri_full = "http://username@example.com:123/path/data?key=value&key2=value2#fragid1"
        uri_full_invalid = "username@example.com:123/path/data?key=value&key2=value2#fragid1"

        self.assertTrue(_matches(pm.RE_URL, uri_user))
        self.assertTrue(_matches(pm.RE_URL, uri_user_2))
        self.assertTrue(_matches(pm.RE_URL, uri_mail))
        self.assertTrue(_matches(pm.RE_URL, uri_full))
        self.assertFalse(_matches(pm.RE_URL, uri_full_invalid))

        url_ssh_1 = "https://user1@domain.com:user2/file.ext"
        url_ssh_2 = "https://username@hostname/path"
        url_ssh_3 = "ssh://username@hostname:1010/path"

        self.assertTrue(_matches(pm.RE_URL, url_ssh_1))
        self.assertTrue(_matches(pm.RE_URL, url_ssh_2))
        self.assertTrue(_matches(pm.RE_URL, url_ssh_3))

    def test_email(self):
        # E-MAIL
        email_simple = "user101-a_b@mail.com"
        email_sub = "yser@one.two.tld"
        email_dotuser = "user.101@mail.tld"

        self.assertTrue(_matches(pm.RE_EMAIL, email_simple))
        self.assertTrue(_matches(pm.RE_EMAIL, email_sub))
        self.assertTrue(_matches(pm.RE_EMAIL, email_dotuser))

    def test_REMatcher(self):
        s = "Hello World"
        space = pm.RE_SPACE
        nSpace = pm.RE_NOT_SPACE

        rem = pm.REMatcher()

        self.assertTrue(rem.match(nSpace, s))
        s = rem.skip()
        self.assertTrue(rem.match(space, s))
        s = rem.skip()
        self.assertTrue(rem.match(nSpace, s))
        s = rem.skip()

        self.assertEqual(len(s), 0)

    def test_re_or(self):
        re1 = 'A'
        re2 = 'B'

        re_1or2 = pm.re_or([re1, re2])
        self.assertEqual(re_1or2, r'(?:A|B)')

        re_1 = pm.re_or([re1])
        self.assertEqual(re_1, r'(?:A)')

    def test_apostrophes(self):
        s = "don't do that, I havn`t seen it before"
        apos = re.compile(pm.APOSTROPHES, re.UNICODE)
        all_apos = apos.findall(s)
        self.assertEqual(len(all_apos), 2)

    def test_contractions_common(self):
        s = "Don't"

        c = re.compile(pm.re_group(pm.CONTRACTION), re.UNICODE)

        cw = re.compile(pm.CONTRACTION_WORD_SPLIT, re.UNICODE)
        c_found = cw.match(s)
        print(s + " ", c_found.groups())

        s = "Doesn't've"
        c_found = cw.match(s)
        print(s + " ", c_found.groups())

        s = "n't've"
        c_found = c.match(s)
        print(s + " ", c_found.groups())

        s = "He'dn't've done that"
        c_found = cw.match(s)
        print(s + " ", c_found.groups())

    def test_contractions_split(self):
        s1 = "Don't do that again I'm serious, I'll be right there."
        c_split = re.compile(pm.CONTRACTION_WORDS, re.UNICODE)

        r1 = c_split.findall(s1)
        print(r1)

        s2 = "I'dn't've what you want though and He wouldn't've that either"
        r2 = c_split.findall(s2)
        print(r2)

    def test_contraction_word(self):
        s1 = "Don't"
        s2 = "Shouldn't've"
        s3 = "He'dn't've"
        s4 = "'twas"
        s5 = "'twasn't"
        s6 = "y'all"
        s7 = "Give'em"

        r1 = re.match(pm.CONTRACTION_WORD_1, s1)
        self.assertEqual(len(r1.groups()), 2)
        self.assertTupleEqual(r1.groups(), ("Do", "n't"))

        r2 = re.match(pm.CONTRACTION_WORD_1, s2)
        self.assertEqual(len(r2.groups()), 2)
        self.assertTupleEqual(r2.groups(), ("Should", "n't've"))

        r3 = re.match(pm.CONTRACTION_WORD_1, s3)
        self.assertEqual(len(r3.groups()), 2)
        self.assertTupleEqual(r3.groups(), ("He", "'dn't've"))

        r4 = re.match(pm.CONTRACTION_WORD_2, s4)
        self.assertEqual(len(r4.groups()), 2)
        self.assertTupleEqual(r4.groups(), ("'t", "was"))

        r5 = re.match(pm.CONTRACTION_WORD_2, s5)
        self.assertEqual(len(r5.groups()), 2)
        self.assertTupleEqual(r5.groups(), ("'t", "was"))

        r6 = re.match(pm.CONTRACTION_WORD_3, s6)
        self.assertEqual(len(r6.groups()), 2)
        self.assertTupleEqual(r6.groups(), ("y'", "all"))

        r7 = re.match(pm.WORD, s7)
        self.assertEqual(len(r7.groups()), 0)
        self.assertEqual(r7.group(), "Give")





    def test_number_patterns(self):
        num_pattenrs = {
            'ISODATE': pm.ISO8601DATETIME,
            'DATE': pm.DATE,
            'PHONE': pm.PHONE_LIKE,
            'FRACTION': pm.LIKELY_FRACTIONS,
            'TIME': pm.TIME,
            'RATIO': pm.RATIO,
            'NUMBER': pm.NUMBER,
            'VERSION': pm.VERSION,
            'SUBSUP_NUMBER': pm.SUBSUP_NUMBER,
            'FRACTION_2': pm.VULGAR_FRACTIONS
        }

        time = "03:40:30"
        ratio = "1:4"
        not_ratio = "01:4"
        date_1 = "27-02-2000"
        date_2 = "02/02/90"
        date_iso = "2016-10-01T16:15:35Z"
        version_1 = "1.2.1"
        version_2 = "v1.0.3"
        number_1 = "-1.2"
        number_2 = "1.2000,39"
        number_3 = "1,2000.39"
        phone_1 = "776-2323"
        phone_2 = "+21 21 928 239"

        # Debugging purposes (some patterns overlap)
        debug = False
        if debug:
            print_matches(num_pattenrs, time)
            print_matches(num_pattenrs, ratio)
            print_matches(num_pattenrs, not_ratio)
            print_matches(num_pattenrs, date_1)
            print_matches(num_pattenrs, date_2)
            print_matches(num_pattenrs, version_1)
            print_matches(num_pattenrs, version_2)
            print_matches(num_pattenrs, number_1)
            print_matches(num_pattenrs, number_2)
            print_matches(num_pattenrs, number_3)
            print_matches(num_pattenrs, phone_1)
            print_matches(num_pattenrs, phone_2)


        date_1_match = re.fullmatch(pm.DATE,date_1)
        self.assertTrue(date_1_match is not None)

        date_2_match = re.fullmatch(pm.DATE,date_2)
        self.assertTrue(date_2_match is not None)

        # numeric entity matching order matters
        # we want to match each example exactly
        # a submatch means there is some conflict
        m = re.match(pm.NUMERIC_ENTITY,time)
        self.assertEqual(time, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY,ratio)
        self.assertEqual(ratio, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY,not_ratio)
        self.assertEqual(not_ratio, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, date_1)
        self.assertEqual(date_1, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, date_2)
        self.assertEqual(date_2, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, date_iso)
        self.assertEqual(date_iso,m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, version_1)
        self.assertEqual(version_1, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, version_2)
        self.assertEqual(version_2, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, number_1)
        self.assertEqual(number_1, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, number_2)
        self.assertEqual(number_2, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, number_3)
        self.assertEqual(number_3, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, phone_1)
        self.assertEqual(phone_1, m.group(0))

        m = re.match(pm.NUMERIC_ENTITY, phone_2)
        self.assertEqual(phone_2, m.group(0))




