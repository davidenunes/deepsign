import math


def subsamplig_prob_cut(word_freq, total_freq, threshold=math.pow(10, -4)):
    freq_cut = threshold * total_freq

    if word_freq < freq_cut:
        return 0.0
    else:
        return 1 - math.sqrt(freq_cut / word_freq)
