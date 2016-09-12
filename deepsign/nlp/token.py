import unicodedata
import re


def is_punct(token):
    for c in token:
        if not unicodedata.category(c).startswith('P'):
            return False

    return True


def is_bracket(token):
    # Penn Treebank bracket tokens
    ptb_brackets = ('-LRB-', '-RRB-', '-RSB-', '-RSB-', '-LCB-', '-RCB-')
    brackets = ('(', ')', '[', ']', '{', '}', '<', '>',) + ptb_brackets

    return token in brackets


def is_quote(token):
    quotes = ('"', "'", '`', '«', '»', '‘', '’', '‚', '‛', '“', '”', '„', '‟', '‹', '›', '❮', '❯', "''", '``')
    return token in quotes


def is_left_punct(token):
    left_punct = ('(', '[', '{', '<', '"', "'", '«', '‘', '‚', '‛', '“', '„', '‟', '‹', '❮', '``')
    return token in left_punct


def is_right_punct(token):
    right_punct = (')', ']', '}', '>', '"', "'", '»', '’', '”', '›', '❯', "''")
    return token in right_punct


# URLs
def _regex_or(*items):
    return '(?:' + '|'.join(items) + ')'


_punct_chars = r"['\"“”‘’.?!…,:;]"
_entity = r"&(?:amp|lt|gt|quot);"

_url_start1 = r"(?:https?://|\bwww\.)"
_common_TLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
_cc_TLDs = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
           r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
           r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
           r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
           r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
           r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
           r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
           r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"
_url_start2 = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + _regex_or(_common_TLDs,
                                                                         _cc_TLDs) + r"(?:\." + _cc_TLDs + r")?(?=\W|$)"
_url_body = r"(?:[^\.\s<>][^\s<>]*?)?"
_url_crap_before_end = _regex_or(_punct_chars, _entity) + "+?"
_url_end = r"(?:\.\.+|[<>]|\s|$)"
_url = _regex_or(_url_start1, _url_start2) + _url_body + "(?=(?:" + _url_crap_before_end + ")?" + _url_end + ")"
_url_re = re.compile(_url)


def is_url(token):
    return _url_re.match(token) is not None


# E-MAIL
_email_bound = r"(?:\W|^|$)"
_email = _regex_or("(?<=(?:\W))",
                   "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" + _email_bound + ")"

_email_re = re.compile(_email)


def is_email(token):
    return _email_re.match(token) is not None
