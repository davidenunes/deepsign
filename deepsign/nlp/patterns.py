"""
DeepSign regular expression patterns

These regular expressions were inspired in (and improved from) expressions used in different tokenizers including:

    * Stanford CoreNLP PTB Lexer: https://github.com/stanfordnlp/CoreNLP/blob/f9c5b58184401bd6db177b76aacefcb749c35a03/src/edu/stanford/nlp/process/PTBLexer.flex
    * Ark Twitter Tokenizer: https://github.com/myleott/ark-twokenize-py/blob/master/twokenize.py
    * SegTok Tokenizer: https://github.com/fnl/segtok/blob/master/segtok/tokenizer.py
    * NLTK PTB Tokenizer: https://github.com/nltk/nltk/blob/develop/nltk/tokenize/treebank.py
    * NLTK Tweet Tokenizer: http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
"""

# Patterns:
# TODO punctuation
# TODO emoji
# TODO arrows and decors
# TODO abreviations
import re


# utility fn and classes
def re_not(pattern):
    return r'(?!'+pattern+')'


def re_or(patterns):
    return r'(?:%s)' % "|".join(patterns)


def re_group(pattern):
    return "("+pattern+")"

def re_boundary_s(pattern):
    return r'(?:(?<=(?:'+pattern+')) | (?<=(?:^)))'


def re_boundary_e(pattern):
    return r'(?='+pattern+'|$)'


def compile(pattern):
    re.compile(pattern, re.UNICODE)


class REMatcher:
    """
    RegEx Matcher utility class that applies match to a string
    and saves the state of that match so that we can re-use it
    in the tokenizer
    """

    def match(self, re_pattern, text):
        """
        Returns True if the pattern matches input at the begining of the string
        False otherwise

        :param re_pattern: the compiled re pattern to be matched in the input
        :param input: the input where we look for a pattern
        """
        m = re_pattern.match(text)
        if m is not None:
            self.input = text
            self.matched = m
            return True
        return False

    def skip(self):
        (_, i) = self.matched.span()
        return self.input[i:]

# punctuation patterns

SPACE = r'\s'
SPACES = r'\s+'
NOT_SPACE = r'[^\s]+'

APOSTROPHE = r'[\'\u201B\u02BC\u2018\u2019\u2032\u0091\u0092\u0060]'
HYPHEN = r'[\-_\u058A\u2010\u2011]'




# didn't include all the symbols just the ones I thought it could appear
# starting parenthesis and brackets
PARENS_BRACKET_S = r'[\(\[\{\u2329\u2768\u2E28\u3008\u300A\uFE59\uFE5B\uFF08\uFF3B\uFF5B]'
# ending parenthesis and brackets
PARENS_BRACKET_E = r'[\)\]\}\u232A\u2769\u2E29\u3009\u300B\uFE5A\uFE5C\uFF09\uFF3D\uFF5D]'
# any parenthesis or brackets
PARENS_BRACKET = re_or([PARENS_BRACKET_S, PARENS_BRACKET_E])


# ubiquitous quote that might appear anywhere
QUOTE_U = re_or([APOSTROPHE,r'\'{2}',r'\"'])
# starting quotes
QUOTE_S = r'[\u2018\u201C\u0091\u0093\u2039\u00AB\u201A\u201E]{1,2}'
# ending quotes
QUOTE_E = r'[\u2019\u201D\u0092\u0094\u203A\u00BB\u201B\u201F]{1,2}'

# any quote
QUOTES = re_or([
    QUOTE_U,
    QUOTE_S,
    QUOTE_E
])

# Stanford PTBLexer includes caracters for armenian arabic, etc
# I'm trying to tokenize regular english perhaps I can include such
# chars in the future but not now. I removed ideographic, arabic, armenian
# etc, commas, question marks and so on
#
# I did include small and fullwidth punctuation (dunno where to expect that though)
DOTS = r'\.\.\.+|[\u0085\u2025\u2026]|\.[ \u00A0](?:\.[ \u00A0])+\.'

# punctuation except hyphens, paranthesis and quotes
PUNCT_INSIDE = r'[,;:\u204F\uFE50\uFE54\uFE55\uFF1B\uFF0C\uFF1A\uFF1B]'
PUNCT_END = r'[\.?!¡¿\u2047\u2048\2049\uFE52\uFE56\uFE57\uFF01\uFF0E\uFF1F]'


PUNCT = re_or([PUNCT_INSIDE,PUNCT_END,HYPHEN,DOTS])
PUNCT_SEQ = PUNCT+'+'


# Alpha num. patterns

DIGIT = r'\d'

U_NUMBER = r'\d+|\d*(?:[.:,\u00AD\u066B\u066C]\d+)+'
S_NUMBER = r'[\-+]?' + U_NUMBER
SUBSUP_NUMBER = r'[\u207A\u207B\u208A\u208B]?(?:[\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+|[\u2080-\u2089]+)'

LIKELY_FRACTIONS = r'(?:\d{1,4}[\- \u00A0])?\d{1,4}(?:\\?\/|\u2044)\d{1,4}'
VULGAR_FRACTIONS = r'[\u00BC\u00BD\u00BE\u2153-\u215E]'

CURRENCY = r'[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0]'

PERCENT = S_NUMBER+'%'

DATE = r'\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}'

TIME_LIKE = r'\d+(?::\d+){1,2}'

ISO8601DATETIME = r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[x0-9]{2}:[0-9]{2}Z?'

DEGREES = r'\u00B0[CF]'

PHONE = r"""
    (
        \([0-9]{2,3}\)[ \u00A0]?
        |
        (\+\+?)?
        ([0-9]{2,4}[\- \u00A0])?
        [0-9]{2,4}[\- \u00A0]
    )
    [0-9]{3,4}[\- \u00A0]?[0-9]{3,5}
    |
    ((\+\+?)?[0-9]{2,4}\.)?
    [0-9]{2,4}\.[0-9]{3,4}\.[0-9]{3,5}
"""

# latin and accented characters
LETTER_ACCENT = r'(?i)(?:(?![×Þß÷þø])[a-zÀ-ÿ])'

# Extra unicode letters (from Stanford CoreNLP Lexer), I labeled the ranges so that people know what they are doing
_UNICODE_EXTRA_WORD_CHARS = [
    '\u00AD',                                                                               # soft hyphen
    '\u0237-\u024F',                                                                        # latin small (ȷùęō)
    '\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02FF',                                              # modifier letters (ʷʰˁ˟)
    '\u0300-\u036F',                                                                        # combining accents
    '\u0370-\u037D\u0384\u0385\u03CF\u03F6\u03FC-\u03FF',                                   # greek letters
    '\u0483-\u0487\u04CF\u04F6-\u04FF\u0510-\u0525',                                        # cyrillic letters
    '\u055A-\u055F',                                                                        # armenian›
    '\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7',                                    # hebrew
    '\u0615-\u061A\u063B-\u063F\u064B-\u065E\u0670\u06D6-\u06EF\u06FA-\u06FF\u0750-\u077F', # arabic
    '\u070F\u0711\u0730-\u074F',                                                            # syriac
    '\u07A6-\u07B1',                                                                        # thaana
    '\u07CA-\u07F5\u07FA',                                                                  # nko
    '\u0900-\u0903\u093C\u093E-\u094E\u0951-\u0955\u0962-\u0963',                           # devanagari
    '\u0981-\u0983\u09BC-\u09C4\u09C7\u09C8\u09CB-\u09CD\u09D7\u09E2\u09E3',                # bengali
    '\u0A01-\u0A03\u0A3C\u0A3E-\u0A4F',                                                     # gurmukhi
    '\u0A81-\u0A83\u0ABC-\u0ACF',                                                           # gujarati
    '\u0B82\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD',                                        # tamil
    '\u0C01-\u0C03\u0C3E-\u0C56',                                                           # telugu
    '\u0D3E-\u0D44\u0D46-\u0D48',                                                           # malayalam
    '\u0E30-\u0E3A\u0E47-\u0E4E',                                                           # thai
    '\u0EB1-\u0EBC\u0EC8-\u0ECD',                                                           # lao
]

LETTER_EXTRA = r'['+"".join(_UNICODE_EXTRA_WORD_CHARS)+']'
LETTER = re_or([LETTER_ACCENT,LETTER_EXTRA])

WORD = "{letter}(?:{letter}|{digit})*(?:[.!?]{letter}(?:{letter}|{digit})*)*".format(letter=LETTER, digit=DIGIT)

# Originally Stanford PTB Lexer has a list of abbreviation regexes to match the ones found in WSJ corpus
# I made one a little bit more general to match things I don't want to get separated Ph.D, a.k.a., p.m., etc
ABBREV = LETTER + '{1,4}(?:\.'+LETTER+'{1,4})+'

# v1.0, 0.5.2 there are other valid tags, but this is the most common
VERSION = LETTER+'?\d(?:\.\d)*'

# Contractions
CONTRACTION_1 = "(?:n{apo}t)".format(apo=APOSTROPHE)                               # n't
CONTRACTION_2 = "(?:{apo}(?:[msd]|re|ve|ll))".format(apo=APOSTROPHE)               # 'm 've 'd 'll 're
CONTRACTION_3 = "(?:{apo}t)".format(apo=APOSTROPHE)                                # it -> 'tis 'twas
CONTRACTION_4 = "(?:y{apo})".format(apo=APOSTROPHE)                                # you -> y'all
CONTRACTION = re_or([CONTRACTION_1, CONTRACTION_2, CONTRACTION_3, CONTRACTION_4])

CONTRACTION_WORD_1 = '(?i)([a-z]+)' + "({c})".format(c=CONTRACTION)          # Don't I'm You're He'll He's
CONTRACTION_WORD_2 = "(?i)({c3})(is)|({c3})(was)".format(c3=CONTRACTION_3)   #'tis 'twas
CONTRACTION_WORD_3 = "(?i)({c4})(all)".format(c4=CONTRACTION_4)              # y'all
CONTRACTION_WORD = re_or([
    CONTRACTION_WORD_1,
    CONTRACTION_WORD_2,
    CONTRACTION_WORD_3
])

# TODO also we have to deal with words that start with apostrophe
# there are two kinds: the ones that come from contractions after
# we parse the sentence and skip tokens like words at the begining of words
# and sentences surrounded by things that can be used as apostrophes as quotes

# TODO other than Contractions we can have assimilations that could be split
# e.g. cannot, gimme, gonna, shoulda

# Extra words not to be separated ---some from Stanford PTBLexer
CONTRACTION_WORD_EXTRA = r"""(?i)
    {apo}n{apo}?|
    {apo}?k|
    {apo}em|
    {apo}im|
    {apo}er|
    {apo}till?|
    {apo}sup|
    {apo}cause|
    {apo}cuz|
    {apo}[2-9]0s|
    [A-HJ-XZn]{apo}{letter}{letter}{letter}*|
    {letter}{letter}*[aeiouy]{apo}[aeiouA-Z]{letter}*|
    [ldj]{apo}|
    dunkin{apo}|
    somethin{apo}|
    ol{apo}|
    o{apo}clock|
    cont{apo}d\.?|
    nor{apo}easter|
    c{apo}mon|
    e{apo}er|
    s{apo}mores|
    ev{apo}ry|
    li{apo}l|
    cap{apo}n""".format(apo=APOSTROPHE,letter=LETTER)

# original regex from @gruber https://gist.github.com/winzig/8894715
# just added the optional port number
URL = r"""
(?i)
\b
(							            # Capture 1: entire matched URL
  (?:
    (?:https?:|[A-z]{2,}:)              # URL protocol and colon
    (?:
      /{1,3}						    # 1-3 slashes
      |								    #   or
      [a-z0-9%]						    # Single letter or digit or '%'
    )                                   # (Trying not to match e.g. "URI::Escape")
    |							        #   or
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:							        # One or more:
    [^\s()<>{}\[\]]+					# Run of non-space, non-()<>{}[]
    |								    #   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
  )+
  (?:							        # End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
    |									#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]		# not a space or one of these punct chars
  )
  |					                    # OR, the following to match naked domains:
  (?:
    (?<!@)			                    # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    (?::\d{2,5})?                       # optional port
    \b
    /?
    (?!@)			                    # not succeeded by a @, avoid matching "foo.na" in "foo.na@example.com"
  )
)
"""

EMAIL = r"""
    (?:(?<=(?:\W))|(?<=(?:^)))                          # e-mail boundary
    (
        [a-zA-Z0-9.\-_']+                                # user
        @
        \b(?:[A-Za-z0-9\-])+(?:\.[A-Za-z0-9\-]+)*         # host
        \.
        (?:\w{2,})                                      # TLD
    )
    (?=\W|$)                                            # e-mail boundary
"""

TWITTER_HANDLE = r'[@\uFF20][a-zA-Z_][a-zA-Z_0-9]*'
HASHTAG = r'[#\uFF03][\w_]+[\w\'_\-]*[\w_]+'

"""
EMOTICONS =
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )
"""


_emote_eyes = r'[:=;]'                      # eyes ---8 and x cause problems
_emote_nose = r'(?:|-|[^a-zA-Z0-9 ]|[Oo])'  # nose ---doesn't get :'-(
_emote_mouth_happy = r'[D\)\]\}]+'
_emote_mouth_sad = r'[\(\[\{]+'
_emote_mouth_tongue = r'[pPd3]+(?=\W|$|RT|rt|Rt)'
_emote_mouth_other = r'(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)'

EMOTICON_STANDARD = _emote_eyes + _emote_nose + \
                    re_or([_emote_mouth_happy,
                           _emote_mouth_sad,
                           _emote_mouth_tongue,
                           _emote_mouth_other])

EMOTICON_REVERSED = re_boundary_s(SPACE) + \
                    re_or([_emote_mouth_happy,
                           _emote_mouth_sad,
                           _emote_mouth_other])

# TODO east emote styles like (T_T)

EMOTICON = re_or([
    EMOTICON_STANDARD,
    EMOTICON_REVERSED
])

HEARTS = "(?:<+/?3+)+"
ARROWS = re_or([r'(?:<*[{hyphen}=]*>+|<+[{hyphen}=]*>*)', '[\u2190-\u21ff]+']).format(hyphen=HYPHEN)

