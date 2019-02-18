from tensorx.data import itertools as itx


def to_ngrams(corpus_fn,
              vocab,
              epochs=1,
              ngram_size=5,
              batch_size=1,
              shuffle=True,
              shuffle_buffer_size=1e5,
              flatten=False,
              enum_epoch=True):
    """ Random N-Gram Pipeline

    Args:
        enum_epoch: if true returns the epoch index (0 batch_1) (0 batch_2) (0 batch_3) ... (1 batch_1)
        shuffle_buffer_size: buffer size for shuffle
        corpus_fn: callable that returns an iterator over the data
        epochs: number of epochs the dataset is repeated for
        vocab: dictionary that converts strings to word ids
        ngram_size: the size of the n-gram window
        batch_size: batch size for the n-gram batch
        shuffle: if true, shuffles the n-grams according to a buffer size
        flatten: if true sliding windows are applied over a stream of words rather than within each sentence
        (n-grams can cross sentence boundaries)
    """

    def pipeline(corpus_stream):
        if flatten:
            word_it = itx.flatten_it(corpus_stream)
            n_grams = itx.window_it(word_it, ngram_size)
        else:
            sentence_n_grams = (itx.window_it(sentence, ngram_size) for sentence in corpus_stream)
            n_grams = itx.flatten_it(sentence_n_grams)

        # at this point this is an n_gram iterator
        n_grams = ([vocab[w] for w in ngram] for ngram in n_grams)

        if shuffle:
            n_grams = itx.shuffle_it(n_grams, shuffle_buffer_size)

        n_grams = itx.batch_it(n_grams, size=batch_size, padding=False)
        return n_grams

    return itx.repeat_apply(lambda fn: pipeline(fn()),
                            corpus_fn,
                            n=epochs,
                            enum=enum_epoch)


def to_parallel_seq(corpus_fn,
                    vocab,
                    epochs=1,
                    seq_len=20,
                    seq_prob=0.95,
                    min_seq_len=5,
                    batch_size=1,
                    num_batches_buffer=None,
                    enum_seq=False,
                    enum_epoch=True,
                    return_future=True):
    """ Parallel Sequence Pipeline for sequence modelling with RNN networks

    Args:
        enum_seq: if specified returns the id for the given parallel sequence
        num_batches_buffer: if not None buffers the given number of batches
        min_seq_len: minimum sequence length
        seq_prob: base sequence probability seq_len // 2 sequences are created with prob. 1-seq_prob
        enum_epoch: if true returns the epoch index (0 batch_1) (0 batch_2) (0 batch_3) ... (1 batch_1)
        corpus_fn: callable that returns an iterator over the data
        epochs: number of epochs the dataset is repeated for
        vocab: dictionary that converts strings to word ids
        seq_len: the size of base sequence
        batch_size: batch size for the n-gram batch
    """

    def pipeline(corpus_stream):
        word_it = itx.flatten_it(corpus_stream)
        ids_it = ([vocab[w] for w in word_it])
        seq_batches = itx.bptt_it(seq=ids_it,
                                  batch_size=batch_size,
                                  seq_len=seq_len,
                                  min_seq_len=min_seq_len,
                                  seq_prob=seq_prob,
                                  num_batches=num_batches_buffer,
                                  enum=enum_seq,
                                  return_targets=return_future
                                  )

        return seq_batches

    batch_it = itx.repeat_apply(lambda fn: pipeline(fn()),
                                corpus_fn,
                                n=epochs,
                                enum=enum_epoch)

    return batch_it
