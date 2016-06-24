import collections

class WordWindow():
    """ A word window contains:
        a left [] of strings
        a word which is in the center of the window
        a right [] of strings
    """
    def __init__(self,left,word,right):
        self.left = left
        self.word = word
        self.right = right

    def __str__(self):
        return "("+str(self.left)+","+self.word+","+str(self.right)+")"


def sliding_windows(tokens, window_size=1):
    """ converts an array of strings to a sequence of windows

    :param tokens: a sequence of strings
    :param window_size: the size of the window around each word
    :return: an array of WordWindow instances
    """
    word_indexes = range(0,len(tokens))
    num_tokens = len(tokens)

    windows = []
    # create a sliding window for each word
    for w in word_indexes:

        # lower limits
        wl = max(0,w - window_size)
        wcl = w

        # upper limits
        wch = num_tokens if w == num_tokens-1 else min(w+1, num_tokens-1)
        wh = w + min(window_size + 1, num_tokens)

        # create window
        left = tokens[wl:wcl]
        word = tokens[w]
        right = tokens[wch:wh]

        windows.append(WordWindow(left,word,right))

    return windows

