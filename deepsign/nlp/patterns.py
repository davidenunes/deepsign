"""
DeepSign regular expression patterns

These regular expressions were inspired in (and improved from) expressions used in different tokenizers including:

    * Stanford CoreNLP PTB Lexer: https://github.com/stanfordnlp/CoreNLP/blob/f9c5b58184401bd6db177b76aacefcb749c35a03/src/edu/stanford/nlp/process/PTBLexer.flex
    * Ark Twitter Tokenizer: https://github.com/myleott/ark-twokenize-py/blob/master/twokenize.py
    * SegTok Tokenizer: https://github.com/fnl/segtok/blob/master/segtok/tokenizer.py
    * NLTK PTB Tokenizer: https://github.com/nltk/nltk/blob/develop/nltk/tokenize/treebank.py
    * NLTK Tweet Tokenizer: http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
"""

# utility fn and classes
from deepsign.nlp.regex_utils import re_or, re_boundary_s, re_boundary_e

# punctuation patterns

SPACE = r'\s'
SPACES = r'\s+'
NOT_SPACE = r'[^\s]+'

APOSTROPHE = r'[\'\u201B\u02BC\u2018\u2019\u2032\u0091\u0092\u0060]'
HYPHEN = r'[\-_\u058A\u2010-\u2015]'

# didn't include all the symbols just the ones I thought it could appear
# starting parenthesis and brackets
PARENS_BRACKET_S = r'[\(\[\{\u2329\u2768\u2E28\u3008\u300A\uFE59\uFE5B\uFF08\uFF3B\uFF5B]'
# ending parenthesis and brackets
PARENS_BRACKET_E = r'[\)\]\}\u232A\u2769\u2E29\u3009\u300B\uFE5A\uFE5C\uFF09\uFF3D\uFF5D]'
# any parenthesis or brackets
PARENS_BRACKET = re_or([PARENS_BRACKET_S, PARENS_BRACKET_E])

# ubiquitous quote that might appear anywhere
QUOTE_U = re_or([APOSTROPHE, r'\'{2}', r'\"'])
# starting quotes
QUOTE_S = r'[\u2018\u201C\u0091\u0093\u2039\u00AB\u201A\u201E]{1,2}'
# ending quotes
QUOTE_E = r'[\u2019\u201D\u0092\u0094\u203A\u00BB\u201B\u201F]{1,2}'

# any quote
QUOTE = re_or([
    QUOTE_U,
    QUOTE_S,
    QUOTE_E
])

# Stanford PTBLexer includes caracters for armenian arabic, etc
# I'embed_size trying to tokenize regular english perhaps I can include such
# chars in the future but not now. I removed ideographic, arabic, armenian
# etc, commas, question marks and so on
#
# I did include small and fullwidth punctuation (dunno where to expect that though)

PUNCT_FN = r'[\\\/@#_\*&\=\+%<>~\*^\|]'

DOTS = r'(?:\.\.\.|[\u0085\u2025\u2026]|\.\s\.\s\.)'

# punctuation except hyphens, paranthesis and quotes
PUNCT_INSIDE = r'[,;:\u204F\uFE50\uFE54\uFE55\uFF1B\uFF0C\uFF1A\uFF1B]'
PUNCT_END = r'\?!|[\.\?!¡¿\u2047\u2048\2049\uFE52\uFE56\uFE57\uFF01\uFF0E\uFF1F]'

PUNCT = re_or([PUNCT_INSIDE, QUOTE + '+', PARENS_BRACKET, DOTS, PUNCT_END, HYPHEN + '+', PUNCT_FN])
PUNCT_SEQ = PUNCT + '+'

# AlphaNum Patterns

DIGIT = r'\d'

# don't care about comma or dot separator consistency
# also avoids confusion with no. 29 and no .29
U_NUMBER = r'(?:' + re_boundary_s('(?![Nn]o?)') + r'\.)?\d+(?:[.,\u00AD]\d+)*(?:[.,\u00AD]\d+)?'
S_NUMBER = r'[\-+]' + U_NUMBER
NUMBER = r'[\-+]?' + U_NUMBER
SUBSUP_NUMBER = r'[\u207A\u207B\u208A\u208B]?(?:[\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+|[\u2080-\u2089]+)'

LIKELY_FRACTIONS = r'(?:\d{1,4}[\- \u00A0])?\d{1,4}(?:\\?\/|\u2044)\d{1,4}'
VULGAR_FRACTIONS = r'[\u00BC\u00BD\u00BE\u2153-\u215E]'

CURRENCY = r'[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]'

PERCENT = NUMBER + '%'
RATIO = r'(?:(?:[1-9]\d+)|\d)(?::\d+)'

DATE_1 = r'[0-9]{1,2}[\-\/][0-9]{1,2}[\-\/][0-9]{2,4}'
DATE_2 = r'[0-9]{2,4}[\-\/][0-9]{1,2}[\-\/][0-9]{1,2}'
DATE = re_or([DATE_1, DATE_2])

TIME = r'\d+(?::\d+){1,2}'

ISO8601DATETIME = DATE_2 + 'T' + TIME + 'Z'

DEGREES = r'\u00b0[CF]'

# added [^\d] because versions don't appear integrated in other patterns except for numbers
VERSION = r'[vV]?\d(?:\.\d)*(?:\.\d|\.x)' + re_boundary_e("[^\d]")

PHONE_LIKE = r'[\+]?(?:\(\d{2,3}\)[\s\-])?\d{2,3}(?:[\s\-]\d{2,4})+'

# order sensitive expression to match expressions
# with numbers without conflict
# e.g. date can be part of ISO8601DATE so we check for that first
# useful for numeric entity tokenization
NUMERIC = re_or([
    ISO8601DATETIME,
    DATE,
    PHONE_LIKE,
    LIKELY_FRACTIONS,
    TIME,
    RATIO,
    VERSION,
    NUMBER,
    SUBSUP_NUMBER,
    VULGAR_FRACTIONS
])

# latin and accented characters
LETTER_NORMAL = r'(?i)[A-Za-z]'
LETTER_ACCENT = r'(?i)(?:(?![×Þß÷þø])[a-zÀ-ÿ])'

# TODO using these was causing some words were tacking forever to be matched
# Extra unicode letters (from Stanford CoreNLP Lexer), I labeled the ranges so that people know what they are doing
_UNICODE_EXTRA_WORD_CHARS = [
    '\u0237-\u024F',  # latin small (ȷùęō)
    '\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02FF',  # modifier letters (ʷʰˁ˟)
    '\u0300-\u036F',  # combining accents
    '\u0370-\u037D\u0384\u0385\u03CF\u03F6\u03FC-\u03FF',  # greek letters
    '\u0483-\u0487\u04CF\u04F6-\u04FF\u0510-\u0525',  # cyrillic letters
    '\u055A-\u055F',  # armenian›
    '\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7',  # hebrew
    '\u0615-\u061A\u063B-\u063F\u064B-\u065E\u0670\u06D6-\u06EF\u06FA-\u06FF\u0750-\u077F',  # arabic
    '\u070F\u0711\u0730-\u074F',  # syriac
    '\u07A6-\u07B1',  # thaana
    '\u07CA-\u07F5\u07FA',  # nko
    '\u0900-\u0903\u093C\u093E-\u094E\u0951-\u0955\u0962-\u0963',  # devanagari
    '\u0981-\u0983\u09BC-\u09C4\u09C7\u09C8\u09CB-\u09CD\u09D7\u09E2\u09E3',  # bengali
    '\u0A01-\u0A03\u0A3C\u0A3E-\u0A4F',  # gurmukhi
    '\u0A81-\u0A83\u0ABC-\u0ACF',  # gujarati
    '\u0B82\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD',  # tamil
    '\u0C01-\u0C03\u0C3E-\u0C56',  # telugu
    '\u0D3E-\u0D44\u0D46-\u0D48',  # malayalam
    '\u0E30-\u0E3A\u0E47-\u0E4E',  # thai
    '\u0EB1-\u0EBC\u0EC8-\u0ECD',  # lao
]

SOFT_HYPHEN = r'\u00AD'

LETTER_EXTRA = r'[' + "".join(_UNICODE_EXTRA_WORD_CHARS) + ']'
LETTER = re_or([LETTER_ACCENT, SOFT_HYPHEN])

WORD = "{letter}(?:{letter}|\d)*".format(letter=LETTER)

# f**k s#$t
WORD_CENSORED = LETTER + '{1,2}' + PUNCT_FN + '{1,}' + LETTER + '{1,3}'

# Originally Stanford PTB Lexer has a list of abbreviation regexes to match the ones found in WSJ corpus
# simple abbrev like U.S, a.k.a., etc., AT&T, A&B
ABBREV_SIMPLE = r'(?:[A-Za-z]\.){2,}|[A-Za-z]{2,3}[\.](?:[A-Za-z]\.?)|[A-Za-z]{1,2}[\.&][A-Za-z]|[A-Za-z]\.'

# included some common abbrev.
# having a lexicon is the only way, either that or learn it from corpora
# (anything stop that doesn't separate a sentence can be part of an abbrev)
_ABBREV_EXTRA = [
    r'(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)',  # months
    r'(?:Mon|Tue|Tues|Wed|Thu|Thurs|Fri|Sat|Sun)',  # days
    r'(?:Inc|Cos?|Corp|Pp?t[ye]s?|Ltd|Plc|Rt|Bancorp|Bhd|Assn|Univ|Coll|Intl|Sys|Invt|Elec|Natl|M[ft]g|Dept)',  # co
    r'(?:tel|est|ext|sq|fl|oz)',  # num
    r'(?:Jr|Sr|Bros|Esq|(Ed|Ph)\.D|Ph)',
    r'(?:Mr|Mrs|Ms|Miss|Drs?|Profs?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Capt|Ste?|Col)',
    r'(?:Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|Mt|Ft|Adj|Adv|Asst|Assoc|Ens|Insp|Mlle|Mme|Msgr|Sfc)',
    r'(?:etc|al|seq|acc|cf)',
    r'(?:adj|adv|Anniv|Attrib|Cal|Calc|Civ|Cl|dict)',
    r'(?:ca|Figs?|prop|art|bldg|prop|pp|op|Conf|def|doc|ed|abbr|abbrv|abbrev)',
    r'(?:Blvd|Rd|Ave|Apt|Ctr|Cts?|Hls?|Lk|Ln|Prt|Trl|Addr|Brit)'
]

ABBREV_EXTRA = r'(?i)' + re_or(_ABBREV_EXTRA) + r'\.'
ABBREV = re_or([ABBREV_SIMPLE, ABBREV_EXTRA])

# things like C# C++ etc, a bit general
PROGRAMMING_LANGUAGES = r'[CF][\-\+#]{1,2}'

# TODO contractions can perhaps be handled more elegantly
# Contractions
CONTRACTION_1 = "(?:ngram_size{apo}t)".format(apo=APOSTROPHE)  # ngram_size't
CONTRACTION_2 = "(?:{apo}(?:[msd]|re|ve|ll))".format(apo=APOSTROPHE)  # 've 'd 'll 're
CONTRACTION = re_or([CONTRACTION_1, CONTRACTION_2])

# same as 2 without 'embed_size
_CONTRACTION_2 = "(?:{apo}(?:[sd]|re|ve|ll))".format(apo=APOSTROPHE)  # 've 'd 'll 're
_CONTRACTION = re_or([CONTRACTION_1, _CONTRACTION_2])

# We have to catch this separately because of the O'Malley's of this world
CONTRACTION_WORD_1 = '(?i)(I)({apo}embed_size)'.format(apo=APOSTROPHE)
CONTRACTION_WORD_2 = '(?i)(y' + APOSTROPHE + ')(' + LETTER + '{3,})'  # y'all y'know

CONTRACTION_WORD_3 = '(?i)([a-z]+)' + "({c}+)".format(c=_CONTRACTION)  # Don't I'embed_size You're He'll He's
CONTRACTION_WORD_4 = "(?i)({apo}t)(is|was)".format(apo=APOSTROPHE)  # 'tis 'twas

SUBJECT_CONTRACTION = '(?i)([a-z]+)' + "({c}+)".format(c=CONTRACTION)

# NOTE: we have to deal with words that start with apostrophe
# there are two kinds: the ones that come from contractions after
# we parse the sentence and skip tokens like words at the begining of words
# and sentences surrounded by things that can be used as apostrophes as quotes

# TODO other than Contractions we can have Assimilations that could be split
# e.g. cannot, gimme, gonna, shouldda, wanna, coulda

# Extra words not to be separated ---some from Stanford PTBLexer
_CONTRACTION_WORD_EXTRA = [
    r'{apo}ngram_size{apo}?'.format(apo=APOSTROPHE),
    APOSTROPHE + 'k',
    APOSTROPHE + 'em',
    APOSTROPHE + 'im',
    APOSTROPHE + 'er',
    APOSTROPHE + 'till?',
    APOSTROPHE + 'sup',
    APOSTROPHE + 'cause',
    APOSTROPHE + 'cuz',
    APOSTROPHE + '[2-9]0s',
    '[ldj]' + APOSTROPHE,
    'dunkin' + APOSTROPHE,
    'somethin' + APOSTROPHE,
    'ol' + APOSTROPHE,
    'o' + APOSTROPHE + 'clocl',
    'cont' + APOSTROPHE + 'd\.?',
    'nor' + APOSTROPHE + 'easter',
    'c' + APOSTROPHE + 'mon',
    'e' + APOSTROPHE + 'er',
    's' + APOSTROPHE + 'mores',
    'ev' + APOSTROPHE + 'ry',
    'li' + APOSTROPHE + 'l',
    'cap' + APOSTROPHE + 'ngram_size'
]
CONTRACTION_WORD_EXTRA = r'(?i)' + re_or(_CONTRACTION_WORD_EXTRA)

# any other word with an apostrophe O'Neill O'Malley
CONTRACTION_WORD_OTHER = WORD + APOSTROPHE + WORD

# original regex from @gruber https://gist.github.com/winzig/8894715
# just added the optional port number
# the following regex has backtracking problems when it has to match and fails
# but calling re.match should lead to no problems since it
URL = r"""
(?i)
  (?:
    (?:https?:|[A-z]{2,}:)              # URL protocol and colon
    (?:
      /{1,3}						    # 1-3 slashes
      |								    #   or
      [a-z0-9%]						    # Single letter or digit or '%'
    )                                   # (Trying not to match e.g. "URI::Escape")
    |							        #   or
    [a-z0-9.\-]+[\.]
    (?:[a-z]{2,13})
    /
  )
  (?:							        # One or more:
    [^\s()<>{}\[\]]+					# Run of non-space, non-()<>{}[]
    |								    #   or
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
  )+
  (?:							        # End with:
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
    |									#   or
    [^\s`!()\[\]{};:'"\.,<>?«»“”‘’]		# not a space or one of these punct chars
  )
  |					                    # OR, the following to match naked domains:
  (?:
    (?<!@)			                    # not preceded by a @, avoid matching foo@_mail.com_
    (?:[a-z0-9]+\.)*                    # www or other subdomains
    (?:
        [a-z0-9]+(?:\-[a-z0-9]+)*       #domain names possibly separated by hyphen
        (?:\.[a-z0-9]+)*                #more possible subdomains
        (?:[\.](?:[a-z]{2,13}))+        # TLD domains: .com .uk.co .org.uk
        (?::\d{2,5})?                    # optional port
    )
    /?
    (?!@)			                    # not succeeded by a @, avoid matching "foo.na" in "foo.na@example.com"
  )
"""

# I use a list to comment the regex so that the verbose flag is not necessary
# I don't want each pattern to require different flags UNICODE should be the only requirement
EMAIL = r"""
    (?:(?<=(?:\W))|(?<=(?:^)))                          # e-mail boundary
    (
        [a-zA-Z0-9.\-_\']+                              # user
        @
        \b(?:[A-Za-z0-9\-])+(?:\.[A-Za-z0-9\-]+)*       # host
        \.
        (?:\w{2,})                                      # TLD
    )
    (?=\W|$)                                            # e-mail boundary
"""

TWITTER_HANDLE = r'[@\uFF20][a-zA-Z_][a-zA-Z_0-9]*'
HASHTAG = r'[#\uFF03][\w_]+[\w\'_\-]*[\w_]+'

_emote_eyes = r'[:=;]'  # eyes ---8 and x cause problems
_emote_nose = r'(?:|-o?|[^a-zA-Z0-9 ]|[Oo]|\*|\')'  # nose ---doesn't get :'-(
_emote_mouth_happy = r'[D\)\]]+'
_emote_mouth_sad = r'[\(\[\{]+'
_emote_mouth_tongue = r'[pPd3]+(?=\W|$)'
_emote_mouth_other = r'(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+|@)'

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

_eye = r'[\^ˆᵔ＾￣▔─⌒◡~\.・･•ﾟ°゜⊙oO□\-_✯★◕☆\'✧˘❛T´≧><｀｀`≦<μ♡❤\/\\˙⇀↼x=\+\*￢¬→←∂ゝᗒ]'
_arm = r'[<>～\\＼⌒ヽﾉノ／\/╰╯o٩۶｡ﾟdwシΣ∑Σヾ凸]'
_mouth = r'[\.\-―ー－_ωεз³3З∀‿◡ᴗ︶⌣▽\s人ヮ\^ｏ〇ロ︹ヘ︿‸ᗣ]'

# face not inside () e.g. =_=
VERTICAL_EMOTE_1 = r'[\^x=~<>]\.\[\^x=~<>]|[\-\^x=~<>\']_[\-\^x=~<>\']'
# face inside () e.g. (>_<)
VERTICAL_EMOTE_2 = _arm + r'{0,2}\s?\(.{0,3}' + _eye + r'.{0,2}' + _mouth + r'.{0,2}' + _eye + r'.{0,3}\)\s?' + _arm + '{0,2}'
VERTICAL_EMOTE = re_or([VERTICAL_EMOTE_2, VERTICAL_EMOTE_1])

HEARTS = "(?:<+/?3+)+"

EMOTICON = re_or([
    VERTICAL_EMOTE,
    EMOTICON_STANDARD,
    EMOTICON_REVERSED,
    HEARTS
])

ARROWS = re_or([r'(?:<*[{hyphen}=]*>+|<+[{hyphen}=]*>*)', '[\u2190-\u21ff]+']).format(hyphen=HYPHEN)

# copyright, registered, copyleft
COPYRIGHT = r'[\u00A9\u00AE\u2184\u2122]'
