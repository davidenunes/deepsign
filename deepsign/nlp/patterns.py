"""
DeepSign: Tokenizers

RegEx for
 * URL
 * E-MAIL

"""

# Patterns:
# TODO twitter (or other social media) handles (e.g. @dude)
# TODO hashtags (e.g. something)
# TODO time and dates
# TODO emoji
# TODO contractions
import re


def _regex_or(*items):
    return '(?:' + '|'.join(items) + ')'


# RFC3986-like URIs
# also matches e-mail addresses since you can have a user before the host
URL = re.compile(
    r"""
    (?<=[\s(\[\{<"'«‘‛“„‟‹❮`])?                           # url boundary
    (
          (?:https?://|\bwww\.|[A-z]{2,}:(?://)?)?        # other protocol
          (?:[A-Za-z\d-]+@)?                              # optional user
          \b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\.     #required host
          (?:\w{2,})(?=\W|$)                              # required TLD
          (?::\d+)?                                       # optional port
          (?:\/[^?\#\s'">)\]}]+)?                         # optional path
          (?:\?[^\#\s'">)\]}]+)?                          # optional query
          (?:\#[^\s'">)\]}]+)?                           # optional fragment
    )
    (?=[\s)\]\}>"'»’”›❯',]|$)?                            # url boundary
    """,
    re.VERBOSE | re.UNICODE)

# E-MAIL ***************************************************************************************************************
_email_bound = r"(?:\W|^|$)"

EMAIL = re.compile(_regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +
                   _email_bound + ")", re.UNICODE)

# Contractions *********************************************************************************************************

CONTRACTIONS = r"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$"
