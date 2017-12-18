def re_not(pattern):
    return r'(?!' + pattern + ')'


def re_or(patterns):
    return r'(?:%s)' % "|".join(patterns)


def re_group(pattern):
    return "(" + pattern + ")"


def re_boundary_s(pattern):
    return r'(?:(?<=(?:' + pattern + ')) | (?<=(?:^)))'


def re_boundary_e(pattern):
    return r'(?=' + pattern + '|$)'


class REMatcher:
    def __init__(self):
        self.input = None
        self.matched = None

    """
    RegEx Matcher utility class that applies match to a string
    and saves the state of that match so that we can re-use it
    in the tokenizer
    """

    def match(self, pattern, text):
        """
        Returns True if the pattern matches input at the begining of the string
        False otherwise

        :param pattern: the compiled re pattern to be matched in the input
        :param input: the input where we look for a pattern
        """
        m = pattern.match(text)
        if m is not None:
            self.input = text
            self.matched = m
            return True
        return False

    def skip(self, group=0):
        (_, i) = self.matched.span(group)
        return self.input[i:]
