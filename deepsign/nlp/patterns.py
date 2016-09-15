"""
DeepSign regular expression patterns

includes patterns for
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


# RFC3986-like URIs
# also matches e-mail addresses since you can have a user before the host
# not perfect but accepts and splits most cases
URL = re.compile(r"""
    (?:(?<=(?:[^\w\/]))|(?<=(?:^)))                     # boundary !word or part of a path (avoid capturing /f.html)
    (
          (?:https?://|[A-z]{2,}:(?://)?)?              # scheme
          (?:[A-Za-z0-9_-]+@)?                          # optional user
          \b(?:[A-Za-z0-9-])+(?:\.[A-Za-z0-9-]+)*       # required host
          \.
          (?:\w{2,})(?=\W|$)                            # required TLD ---followed by non-word char (?=\W|$)
          (?::\d{2,5})?                                 # optional port
          (?:\/@[A-Za-z0-9_-]+)?                        # optional path with handle (e.g. medium.com/@user)
          (?:                                           # begin optional path
            (?:
                \/[^?\#\s'">\]}.]+
                (?:\.[\w]+)*                            # an end in ext or contain dots (e.g. file.html or /v0.2.3/)
            )+                                          # it might occur more than once (/v0.2.1/file.html)
          )?                                            # end optional path
          (?:\?[^\#\s'">)\]}]+)?                        # optional query
          (?:\#[^\s'">)\]}]+)?                          # optional fragment (I don't allow closing parenthesis in fragments
    )
    (?=\W|$)                                            # url boundary
    """, re.VERBOSE | re.UNICODE)


EMAIL = re.compile(r"""
    (?:(?<=(?:\W))|(?<=(?:^)))                          # e-mail boundary
    (
        [a-zA-Z0-9.-_']+                                # user
        @
        \b(?:[A-Za-z0-9-])+(?:\.[A-Za-z0-9-]+)*         # host
        \.
        (?:\w{2,})                                      # TLD
    )
    (?=\W|$)                                            # e-mail boundary
    """, re.VERBOSE | re.UNICODE)



CONTRACTIONS = r"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$"
