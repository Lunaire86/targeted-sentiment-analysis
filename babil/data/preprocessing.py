#!/usr/bin/env python3
# coding: utf-8

from os.path import abspath, join
import pickle
import re
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from typing import Tuple, Set

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


@dataclass
class LabelTokeniser(Tokenizer):
    def __init__(self, **kwargs):
        defaults = {'filters': '', 'lower': False}
        defaults.update(kwargs)
        super(LabelTokeniser, self).__init__(**defaults)
        self.index_word[0] = '<PAD>'
        self.word_index['<PAD>'] = 0

    def save(self, folder: str):
        basename = f'{self.__class__.__name__}.pickle'
        with open(join(folder, basename), 'wb') as f:
            pickle.dump(self, f)

    def fit_on_texts(self, nested_lists):
        texts = [' '.join(_) for _ in nested_lists]
        super().fit_on_texts(texts)

    def texts_to_sequences(self, nested_lists):
        texts = [' '.join(_) for _ in nested_lists]
        return super().texts_to_sequences(texts)


@dataclass
class WordTokeniser(LabelTokeniser):
    def __init__(self, **kwargs):
        defaults = {'num_words': None, 'oov_token': '<UNK>'}
        defaults.update(kwargs)
        super(WordTokeniser, self).__init__(**defaults)


@dataclass
class Dataset:

    filepath: str
    X: List[List[str]] = field(init=False)
    y: List[List[str]] = field(init=False)

    def __post_init__(self):
        conll = open(abspath(self.filepath), 'r')
        # strip removes trailing newlines at the bottom of the document
        text = conll.read().rstrip()
        self.X, self.y = self._parse(text)

        conll.close()

    def _split_Xy(self, text: str) -> List[List[str]]:
        # Split on double newlines for individual sentences
        sentences = text.split('\n\n')

        # 98% of everything I do is overkill,
        # but I couldn't resist me some regex & chill
        p = re.compile(r'(?:(\S+)(?:\t)(\S+))+')

        return [re.findall(p, _) for _ in sentences]

    def _parse(self, text: str) -> Tuple[
        List[List[str]], List[List[str]]
    ]:
        sentences = self._split_Xy(text)

        X, y = [], []
        for sentence in sentences:
            tokens, labels = zip(*sentence)
            X.append(list(tokens))
            y.append(list(labels))
        return X, y


def vectorise(texts: List[List[str]],
              tokeniser: Optional[Union[LabelTokeniser, WordTokeniser]] = None,
              word2idx: Optional[Dict[str, int]] = None,
              categorical: Optional[bool] = False,
              maxlen: int = 50) -> np.ndarray:

    vectorised: List[List[int]] = []
    sequences: List[List[int]] = []
    padded: np.ndarray

    # Case: not using pre-trained word embeddings
    if tokeniser:
        sequences = tokeniser.texts_to_sequences(texts)

    # Case: using pre-trained word embeddings
    if word2idx:
        for sentence in texts:
            # Replace unknown tokens with 1
            encoded = [
                word2idx[word]
                if word in word2idx and word2idx[word] < 20000
                # if word in word2idx  -- TODO : change back
                else 1
                for word in sentence
            ]
            vectorised.append(encoded)

    # Pad all sequences
    padded = pad_sequences(
        sequences or vectorised,
        maxlen=maxlen,
        padding='post',
        truncating='post'
    )

    # Case categorical: vectorising labels -> one-hot encode
    return to_categorical(padded) if categorical else padded


def masking():
    '''
    mask: Binary tensor of shape [batch, timesteps] indicating whether
    a given timestep should be masked
    '''
    # TODO : look into this
    pass
