import unittest

import deepsign.utils.regex
from deepsign.nlp import patterns as pm
from deepsign.nlp import token
from segtok.tokenizer import web_tokenizer
import re


def _matches(pattern,text):
    match = pattern.match(text)

    if match is not None:
        matched = match.group(0)
        return matched == text
    else:
        return False


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

        url_pattern = re.compile(pm.URL, re.VERBOSE | re.UNICODE)


        self.assertFalse(_matches(url_pattern, reject_url_simple_sub))
        self.assertTrue(_matches(url_pattern, url_simple_port))
        self.assertTrue(_matches(url_pattern, url_protocol_port))
        self.assertTrue(_matches(url_pattern, url_simple_path1))
        self.assertTrue(_matches(url_pattern, url_simple_protocol_path1))
        self.assertTrue(_matches(url_pattern, url_simple_path2))
        self.assertTrue(_matches(url_pattern, url_simple_protocol_path2))
        self.assertTrue(_matches(url_pattern, url_simple_path3))
        self.assertTrue(_matches(url_pattern, url_simple_protocol_path3))

        url_scheme_simple = "https://domain.tld"
        url_scheme_sub = "https://sub.domain.tld"
        url_path_asterisk = "http://domain.tld/path/*/http://google.com"
        url_path_bang = "http://domain.tld/path/!/http://google.com"
        url_parenthesis_invalid_1 = "http://example.com/file%2811%281.html)"
        url_parenthesis_2 = "http://example.com/@user/file(1).html"
        url_parenthesis_3 = "http://sub.domain.domain/p1/r2.4.1/file.html#segment(string)"

        self.assertTrue(_matches(url_pattern, url_scheme_simple))
        self.assertTrue(_matches(url_pattern, url_scheme_sub))
        self.assertTrue(_matches(url_pattern, url_path_asterisk))
        self.assertTrue(_matches(url_pattern, url_path_bang))

        self.assertFalse(_matches(url_pattern, url_parenthesis_invalid_1))
        self.assertTrue(_matches(url_pattern, url_parenthesis_2))
        self.assertTrue(_matches(url_pattern, url_parenthesis_3))

        url_fragment = "http://domain.tld/path/#hello"
        url_handle = "https://domain.tld/@username"
        url_tilde = "http://domain.tld/~username"

        self.assertTrue(_matches(url_pattern, url_fragment))
        self.assertTrue(_matches(url_pattern, url_handle))
        self.assertTrue(_matches(url_pattern, url_tilde))

        url_path_query_1 = "domain.com/path?key=value&key2=value2"
        url_path_query_2 = "domain.com/path?key=value&key2=value_2"
        url_path_query_3 = "domain.com/path?key=value1+value_2"
        url_path_query_4 = "domain.com/path/!/value1&value2"
        url_path_query_5 = "domain.com/path$?key=value1"
        url_path_query_6 = "domain.com/path?key=value1;value_2"
        url_path_query_7 = "domain.com/path?key=value1%20value_2"

        self.assertTrue(_matches(url_pattern, url_path_query_1))
        self.assertTrue(_matches(url_pattern, url_path_query_2))
        self.assertTrue(_matches(url_pattern, url_path_query_3))
        self.assertTrue(_matches(url_pattern, url_path_query_4))
        self.assertTrue(_matches(url_pattern, url_path_query_5))
        self.assertTrue(_matches(url_pattern, url_path_query_6))
        self.assertTrue(_matches(url_pattern, url_path_query_7))

        uri_user = "http://user:info@host.name.tld"
        uri_user_2 = "user:info@host.name.tld"
        uri_mail = "mailto:user@mail.tld"
        uri_full = "http://username@example.com:123/path/data?key=value&key2=value2#fragid1"
        uri_full_invalid = "username@example.com:123/path/data?key=value&key2=value2#fragid1"

        self.assertTrue(_matches(url_pattern, uri_user))
        self.assertTrue(_matches(url_pattern, uri_user_2))
        self.assertTrue(_matches(url_pattern, uri_mail))
        self.assertTrue(_matches(url_pattern, uri_full))
        self.assertFalse(_matches(url_pattern, uri_full_invalid))

        url_ssh_1 = "https://user1@domain.com:user2/file.ext"
        url_ssh_2 = "https://username@hostname/path"
        url_ssh_3 = "ssh://username@hostname:1010/path"

        self.assertTrue(_matches(url_pattern, url_ssh_1))
        self.assertTrue(_matches(url_pattern, url_ssh_2))
        self.assertTrue(_matches(url_pattern, url_ssh_3))


    def test_email(self):
        # E-MAIL
        email_simple = "user101-a_b@mail.com"
        email_sub = "yser@one.two.tld"
        email_dotuser = "user.101@mail.tld"

        email_pattern = re.compile(pm.EMAIL, re.UNICODE | re.VERBOSE)
        url_pattern = re.compile(pm.URL, re.VERBOSE|re.UNICODE)

        self.assertTrue(_matches(email_pattern, email_simple))
        self.assertFalse(_matches(url_pattern,email_simple))
        self.assertTrue(_matches(email_pattern, email_sub))
        self.assertTrue(_matches(email_pattern, email_dotuser))

    def test_REMatcher(self):
        s = "Hello World"
        space = re.compile(pm.SPACES)
        nSpace = re.compile(pm.NOT_SPACE)

        rem = deepsign.utils.regex.REMatcher()

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

        re_1or2 = deepsign.utils.regex.re_or([re1, re2])
        self.assertEqual(re_1or2, r'(?:A|B)')

        re_1 = deepsign.utils.regex.re_or([re1])
        self.assertEqual(re_1, r'(?:A)')

    def test_apostrophes(self):
        s = "don't do that, I havn`t seen it before"
        apos = re.compile(pm.APOSTROPHE, re.UNICODE)
        all_apos = apos.findall(s)
        self.assertEqual(len(all_apos), 2)

    def test_contractions_common(self):
        s = "Don't"

        c = re.compile(deepsign.utils.regex.re_group(pm.CONTRACTION), re.UNICODE)

        cw = re.compile(pm.CONTRACTION_WORD_1, re.UNICODE)
        c_found = cw.match(s)

        s = "Doesn't've"
        c_found = cw.match(s)

        s = "n't've"
        c_found = c.match(s)

        s = "He'dn't've done that"
        c_found = cw.match(s)


    def test_contractions_split(self):
        s1 = "Don't do that again I'm serious, I'll be right there."
        c_split = re.compile(pm.CONTRACTION_WORD_1, re.UNICODE)

        r1 = c_split.findall(s1)
        self.assertListEqual([("Do","n't"),("I","'m"),("I","'ll")],r1)


        s2 = "I'dn't've what you want though and He wouldn't've that either"
        r2 = c_split.findall(s2)
        self.assertListEqual([("I", "'dn't've"), ("would", "n't've")], r2)


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

    def test_word_version_conflict(self):
        word_pattern = re.compile(pm.WORD, re.UNICODE)
        version_pattern = re.compile(pm.VERSION, re.UNICODE)

        version = "v1.2"

        # this is a problem, version has to be catched first
        self.assertTrue(word_pattern.match(version) is not None)
        self.assertTrue(_matches(version_pattern,version))

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
        version_3 = "1.2.x"
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
        m = re.match(pm.NUMERIC, time)
        self.assertEqual(time, m.group(0))

        m = re.match(pm.NUMERIC, ratio)
        self.assertEqual(ratio, m.group(0))

        m = re.match(pm.NUMERIC, not_ratio)
        self.assertEqual(not_ratio, m.group(0))

        m = re.match(pm.NUMERIC, date_1)
        self.assertEqual(date_1, m.group(0))

        m = re.match(pm.NUMERIC, date_2)
        self.assertEqual(date_2, m.group(0))

        m = re.match(pm.NUMERIC, date_iso)
        self.assertEqual(date_iso,m.group(0))

        m = re.match(pm.NUMERIC, version_1)
        self.assertEqual(version_1, m.group(0))

        m = re.match(pm.NUMERIC, version_2)
        self.assertEqual(version_2, m.group(0))

        m = re.match(pm.NUMERIC, version_3)
        self.assertEqual(version_3, m.group(0))

        m = re.match(pm.NUMERIC, number_1)
        self.assertEqual(number_1, m.group(0))

        m = re.match(pm.NUMERIC, number_2)
        self.assertEqual(number_2, m.group(0))

        m = re.match(pm.NUMERIC, number_3)
        self.assertEqual(number_3, m.group(0))

        m = re.match(pm.NUMERIC, phone_1)
        self.assertEqual(phone_1, m.group(0))

        m = re.match(pm.NUMERIC, phone_2)
        self.assertEqual(phone_2, m.group(0))

    def test_abbreviations(self):
        abbrev_pattern = re.compile(pm.ABBREV, re.UNICODE)

        # simple abbrev
        simple_abbrev = [
            "U.S.",
            "a.k.a.",
            "a.m.",
            "e.g.",
            "U.S.A.",
            "etc.",
            "Mr.",
            "MR.",
            "MRS.",
            "mrs.",
            "Ph.D",
            "a.m",
            "A.M",
            "Brit."
        ]

        for abbrev in simple_abbrev:
            m = abbrev_pattern.match(abbrev)
            self.assertEqual(m.group(0),abbrev)


        self.assertTrue(abbrev_pattern.match("it. ") is None)

        abbrev_end = "U.S.."
        m = abbrev_pattern.match(abbrev_end)
        self.assertEqual("U.S.", m.group(0))

    # TODO finish the vertical emote pattern to match reasonable emoticons
    def test_emoticons(self):
        emoticons_joy = [
            "(* ^ ω ^)",
            "(´∀｀ * )",
            "(-‿‿ -)",
            "o(≧▽≦)o",
            "(o ^▽ ^ o)",
            "(⌒▽⌒)",
            "< (￣︶￣) >",
            "(*⌒―⌒*)",
            "ヽ(・∀・)ﾉ",
            "(´｡• ω •｡`)",
            "(￣ω￣)",
            "(゜ε゜ )",
            "(o･ω･o)",
            "(＠＾－＾)",
            "ヽ(*・ω・)ﾉ",
            "(o_ _)ﾉ彡☆",
            "(^人^)",
            "(o´ ▽ `o)",
            "(*´▽`*)",
            "(´ω｀)",
            "(≧◡≦ )",
            "(o ´∀｀o)",
            "(´• ω •`)",
            "(＾▽＾)",
            "(⌒ω⌒)",
            "d(ﾟ∀ﾟd)",
            "╰(▔∀▔)╯",
            "(─‿‿─)",
            "(* ^ ‿ ^ * )",
            "ヽ(o ^― ^ o)ﾉ",
            "(✯◡✯)",
            "(◕‿◕)",
            "(*≧ω≦ * )",
            "(☆▽☆)",
            "(⌒‿⌒)",
            "＼(≧▽≦)／",
            "⌒(o＾▽＾o)ノ",
            "～('▽^人)",
            "(*ﾟ▽ﾟ *)",
            "(✧∀✧)",
            "(✧ω✧)",
            "ヽ(*⌒▽⌒ * )ﾉ",
            "(´｡• ᵕ •｡`)",
            "( ´ ▽ ` )",
            "(￣▽￣)",
            "╰(*´︶` *)╯",
            "ヽ( >∀ < ☆)ノ",
            "o(≧▽≦)o",
            "(っ˘ω˘ς )",
            "＼(￣▽￣)／",
            "(*¯︶ ¯ *)",
            "＼(＾▽＾)／",
            "٩(◕‿◕)۶",
            "(o˘◡˘o)",
            "\(★ω★ )/",
            "\ ( ^ ヮ^ )/",
            "(〃＾▽＾〃)",
            "(╯✧▽✧)╯",
            "o ( > ω<)o",
            "o( ❛ᴗ❛ )o",
            "｡ﾟ(TヮT)ﾟ｡"
        ]

        emote_pattern = re.compile(pm.VERTICAL_EMOTE)
        for i in range(len(emoticons_joy)):
            emote = emoticons_joy[i]
            match = emote_pattern.match(emote)
            self.assertTrue(match is not None and len(match.group(0))>0)

        emoticons_love = [
            "(ﾉ´з｀)ノ",
            "(♡μ_μ)",
            "(*^^*)♡",
            "(♡-_-♡)",
            "(￣ε￣＠)",
            "ヽ(♡‿♡)ノ",
            "( ´∀｀)ノ～ ♡",
            "(─‿‿─)♡",
            "(´｡• ᵕ •｡`) ♡",
            "(*♡∀♡)",
            "(｡・//ε//・｡)"
            "(´ω｀♡)",
            "( ◡‿◡ ♡)",
            "(◕‿◕)♡",
            "(/▽＼*)｡o○♡"
            "(ღ˘⌣˘ღ)",
            "(♡ﾟ▽ﾟ♡)",
            "(。-ω-)",
            "～(\'▽^人)"
            "(´• ω •`) ♡",
            "(´ε｀ )♡",
            "(´｡• ω •｡`) ♡",
            "( ´ ▽ ` ).｡ｏ♡"
            "╰(*´︶`*)╯♡",
            "(*˘︶˘*).｡.:*♡",
            "(♡˙︶˙♡)",
            "＼(￣▽￣)／♡",
            "(≧◡≦) ♡",
            "(⌒▽⌒)♡",
            "(*¯ ³¯*)♡",
            "(⇀ 3 ↼)",
            "(￣З￣)",
            "(❤ω❤)",
            "(˘∀˘)/(μ‿μ) ❤",
            "(ˆ⌣ˆc)",
            "(´♡‿♡`)",
            "(°◡°♡)"
        ]

        emote_pattern = re.compile(pm.VERTICAL_EMOTE)
        for i in range(len(emoticons_love)):
            emote = emoticons_love[i]
            match = emote_pattern.match(emote)
            #print(match)
            self.assertTrue(match is not None and len(match.group(0)) > 0)


        emoticons_embarassment = [
            "(⌒_⌒;)",
            "(o^ ^o)",
            "(*/ω＼)",
            "(*/。＼)",
            "(*/_＼)",
            "(*ﾉωﾉ)",
            "(o-_-o)",
            "(*μ_μ)",
            "( ◡‿◡ *)",
            "(ᵔ.ᵔ)",
            "(//▽//)",
            "(//ω//)",
            "(ノ*ﾟ▽ﾟ*)",
            "(*^.^*)",
            "(*ﾉ▽ﾉ)",
            "(￣▽￣*)ゞ",
            "(⁄ ⁄•⁄ω⁄•⁄ ⁄)",
            "(*/▽＼*)",
            "(⁄ ⁄>⁄ ▽ ⁄<⁄ ⁄)"
        ]

        emoticons_negative = [
            "(＃＞＜)", "(；⌣̀_⌣́)", "☆ｏ(＞＜；)○", "(￣ ￣|||)\n(；￣Д￣)", "(￣□￣」)", "(＃￣0￣)", "(＃￣ω￣)\n(￢_￢;)", "(＞ｍ＜)", "(」゜ロ゜)」", "(〃＞＿＜;〃)\n(＾＾＃)", "(︶︹︺)", "(￣ヘ￣)", "<(￣ ﹌ ￣)>\n(￣︿￣)", "(＞﹏＜)", "(--_--)", "凸(￣ヘ￣)\nヾ( ￣O￣)ツ", "(⇀‸↼‶)", "o(>< )o", "(」＞＜)」\n(ᗒᗣᗕ)՞"
        ]

        emoticons_anger = [
            "(＃`Д´)",
            "(｀皿´＃)",
            "(｀ω´)",
            "ヽ( `д´*)ノ",
            "(・｀ω´・)",
            "(｀ー´)",
            "ヽ(｀⌒´メ)ノ",
            "凸(｀△´＃)",
            "(｀ε´)",
            "ψ(｀∇´)ψ",
            "ヾ(｀ヘ´)ﾉﾞ",
            "ヽ(‵﹏′)ノ",
            "(ﾒ｀ﾛ´)",
            "(╬｀益´)",
            "┌∩┐(◣_◢)┌∩┐",
            "凸(｀ﾛ´)凸",
            "Σ(▼□▼メ)",
            "(°ㅂ°╬)",
            "ψ(▼へ▼メ)～→",
            "(ノ°益°)ノ",
            "(҂ `з´ )",
            "(‡▼益▼)",
            "(҂｀ﾛ´)凸",
            "((╬◣﹏◢))",
            "٩(╬ʘ益ʘ╬)۶",
            "(╬ Ò﹏Ó)",
            "＼＼٩(๑`^´๑)۶／／",
            "(凸ಠ益ಠ)凸",
            "↑_(ΦwΦ)Ψ",
            "←~(Ψ▼ｰ▼)∈",
            "୧((#Φ益Φ#))୨",
            "٩(ఠ益ఠ)۶"
        ]

        emoticons_sorrow = [
            "(ノ_<。)", "(*-_-)", "(´-ω-｀)", ".･ﾟﾟ･(／ω＼)･ﾟﾟ･.\n(μ_μ)", "(ﾉД`)", "(-ω-、)", "。゜゜(´Ｏ｀)°゜。\no(TヘTo)", "(；ω；)", "(｡╯3╰｡)", "｡･ﾟﾟ*(>д<)*ﾟﾟ･｡\n( ﾟ，_ゝ｀)", "(个_个)", "(╯︵╰,)", "｡･ﾟ(ﾟ><ﾟ)ﾟ･｡\n( ╥ω╥ )", "(╯_╰)", "(╥_╥)", ".｡･ﾟﾟ･(＞_＜)･ﾟﾟ･｡.\n(／ˍ・、)", "(ノ_<、)", "(╥﹏╥)", "｡ﾟ(｡ﾉωヽ｡)ﾟ｡\n(つω`*)", "(｡T ω T｡)", "(ﾉω･､)", "･ﾟ･(｡>ω<｡)･ﾟ･\n(T_T)", "(>_<)", "(Ｔ▽Ｔ)", "｡ﾟ･ (>﹏<) ･ﾟ｡\no(〒﹏〒)o", "(｡•́︿•̀｡)", "(ಥ﹏ಥ)'"
        ]


        emoticons_fear = [
                  "(ノωヽ)", "(／。＼)", "(ﾉ_ヽ)", "..・ヾ(。＞＜)シ\n(″ロ゛)", "(;;;*_*)", "(・人・)", "＼(〇_ｏ)／\n(/ω＼)", "(/_＼)", "〜(＞＜)〜", "Σ(°△°|||)︴\n(((＞＜)))", "{{ (>_<) }}", "＼(º □ º l|l)/", "〣( ºΔº )〣'"
        ]

        emoticons_confusion = [
            "(￣ω￣;)",
            "σ(￣、￣〃)",
            "(￣～￣;)",
            "(-_-;)",
            "(\'～`;)┌",
            "(・_・ヾ",
            "(〃￣ω￣〃ゞ",
            "┐(￣ヘ￣;)┌",
            "(・_・;)",
            "(￣_￣)・・・",
            "╮(￣ω￣;)╭",
            "(￣.￣;)",
            "(＠_＠)",
            "(・・;)ゞ",
            "Σ(￣。￣ﾉ)",
            "(・・ ) ?",
            "(•ิ_•ิ)?",
            "(◎ ◎)ゞ",
            "(ーー;)",
            "ლ(ಠ_ಠ ლ)"
        ]

        emoticons_doubt = [
            "(￢_￢)",
            "(→_→)",
            "(￢ ￢)",
            "(￢‿￢ )",
            "(¬_¬ )",
            "(←_←)",
            "(¬ ¬ )",
            "(¬‿¬ )",
            "(↼_↼)",
            "(⇀_⇀)"
        ]

        emoticons_surprise = [
            "w(ﾟｏﾟ)w",
            "ヽ(ﾟ〇ﾟ)ﾉ",
            "Σ(O_O)",
            "Σ(ﾟロﾟ)",
            "(⊙_⊙)",
            "(o_O)",
            "(O_O;)",
            "(O.O)",
            "(ﾟロﾟ) !",
            "(o_O) !",
            "(□_□)",
            "Σ(□_□)",
            "∑(O_O;)"
        ]

        emoticons_greeting = [
            "'(*・ω・)ﾉ",
            "(￣▽￣)ノ",
            "(ﾟ▽ﾟ)/",
            "(*´∀｀)ﾉ",
            "(^-^*)/",
            "(＠´ー`)ﾉﾞ",
            "(´• ω •`)ﾉ",
            "(ﾟ∀ﾟ)ﾉﾞ",
            "ヾ(*\'▽\'*)",
            "＼(⌒▽⌒)",
            "ヾ(☆▽☆)",
            "( ´ ▽ ` )ﾉ",
            "(^０^)ノ",
            "~ヾ(・ω・)",
            "(・∀・)ノ",
            "ヾ(^ω^*)",
            "(*ﾟｰﾟ)ﾉ",
            "(・_・)ノ",
            "(o´ω`o)ﾉ",
            "ヾ(☆\'∀\'☆)",
            "(￣ω￣)/",
            "(´ω｀)ノﾞ",
            "(⌒ω⌒)ﾉ",
            "(o^ ^o)/",
            "(≧▽≦)/",
            "(✧∀✧)/",
            "(o´▽`o)ﾉ",
            "(￣▽￣)/"

        ]

        emoticons_winking = [
            "(^_~)",
            "( ﾟｏ⌒)",
            "(^_-)≡☆",
            "(^ω~)",
            "(>ω^)",
            "(~人^)",
            "(^_-)",
            "( -_・)",
            "(^_<)〜☆",
            "(^人<)〜☆",
            "⌒(ゝ。∂)",
            "(^_<)",
            "(^_−)☆",
            "(･ω<)☆"
        ]

        emoticons_apologizing = ["m(_ _)m", "(シ_ _)シ", "m(. .)m", "<(_ _)>\n人(_ _*)", "(*_ _)人", "m(_ _;m)", "(m;_ _)m\n(シ. .)シ"]

