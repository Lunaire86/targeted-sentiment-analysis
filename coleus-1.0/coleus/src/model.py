#!/usr/bin/env python3
# coding: utf-8

from argparse import Namespace
from dataclasses import dataclass
from multiprocessing import cpu_count

from gensim.models.fasttext import FastText
from gensim.models.word2vec import PathLineSentences


@dataclass
class Model(FastText):

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

    @staticmethod
    def init(args: Namespace):
        """Convenient factory method for initialising the model
        using the command-line arguments. This class acts as a
        shallow wrapper around the
        :class:`~gensim.models.fasttext.FastText` class.

        Arguments:
            args: Parsed command-line arguments.
        Returns:
            model: The initialised model.
        """
        init_kwargs = {
            # Dimensionality of the word vectors.
            'size': args.dim,

            # The maximum distance between the current
            # and predicted word within a sentence.
            'window': args.window_size,

            # The model ignores all words with total
            # frequency lower than this.
            'min_count': args.min_count,

            # Train using skip-gram if `sg=1`, else use CBoW.
            'sg': 1 if args.sg else 0,

            # Number of iterations (epochs) over the corpus.
            'iter': args.epochs,

            # Use these many worker threads to train the model
            # (=faster training with multicore machines).
            'workers': min(32, cpu_count()),

            # sort the vocabulary by descending frequency
            # before assigning word indices
            'sorted_vocab': 1
        }
        model = Model(**init_kwargs)
        model.build_vocab(PathLineSentences(args.source))

        return model

    def train(self, *args, **kwargs):
        super(Model, self).train(
            # Path to a corpus file in LineSentence format
            PathLineSentences(args[0]),

            # Count of sentences.
            total_examples=self.corpus_count,

            # Count of raw words in sentences.
            total_words=self.corpus_total_words,

            # Count of words already trained. Set this to 0 for
            # the usual case of training on all words in sentences.
            word_count=0,

            # Number of iterations (epochs) over the corpus.
            epochs=self.epochs
        )
