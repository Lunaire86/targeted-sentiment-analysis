#!/usr/bin/env python3
# coding: utf-8

from multiprocessing import cpu_count
from typing import Union, Any, List

from gensim.models.fasttext import FastText
from gensim.models.word2vec import LineSentence, PathLineSentences


def build(dim: int = 100,
          window: int = 5,
          epochs: int = 5,
          min_count: int = 5,
          neg_sampling: bool = False
          ) -> FastText:

    return FastText(
        size=dim,
        window=window,
        min_count=min_count,
        iter=epochs,
        sorted_vocab=1,
        batch_words=8000,
        hs=0 if neg_sampling else 1,
        workers=min(32, cpu_count()),
        seed=69686
    )


def build_vocab(model: FastText,
                data: Union[LineSentence, PathLineSentences, List[Any, str]]
                ) -> FastText:

    return model.build_vocab(data)


def update_vocab(model: FastText,
                 data: List[Any, str]
                 ) -> FastText:

    return model.build_vocab(
        data,
        update=True
    )


def train(model: FastText,
          data: Union[LineSentence, PathLineSentences, List[Any, str]]
          ) -> FastText:

    return model.train(
        data,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        word_count=0,
        epochs=model.epochs
    )
