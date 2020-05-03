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
    def __init__(self, *args, **kwargs):
        super(LabelTokeniser, self).__init__(filters='', lower=False, *args, **kwargs)

    def save(self, folder: str):
        basename = f'{self.__class__.__name__}.pickle'
        with open(join(folder, basename), 'wb') as f:
            pickle.dump(self, f)


@dataclass
class WordTokeniser(LabelTokeniser):
    def __init__(self, *args, **kwargs):
        super(LabelTokeniser, self).__init__(num_words=None, oov_token='<UNK>', *args, **kwargs)


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

    def _parse(self, text: str) -> Tuple[List[Tuple[str]], List[Tuple[str]]]:
        sentences = self._split_Xy(text)

        X, y = [], []
        for sentence in sentences:
            tokens, labels = zip(*sentence)
            X.append(tokens)
            y.append(labels)
        return X, y



@dataclass
class ConllData:
    """
    Parses a conll-file.
    ---
    methods:
        as_lists:
        - returns a list of lists (sentences) with (token, tag) tuples.
        as_tuples:
        - returns a list of lists (sentencees) with (tokens) and (tags) tuples.
    """
    filepath: str
    _abspath: str = field(init=False)
    _conll_sents: List[str] = field(init=False)
    _parsed_sents: List[List[Tuple[str]]] = field(init=False)
    _X_y_split_sents: List[List[Tuple[str]]] = field(init=False)
    _unique_tokens: Set[str] = field(init=False, default_factory=set)
    _unique_labels: Set[str] = field(init=False, default_factory=set)

    def __post_init__(self):
        self._abspath = os.path.abspath(self.filepath)
        with open(self._abspath, 'r') as f:
            # remove trailing newlines at the bottom of the document
            doc = f.read().rstrip()
        self._conll_sents = doc.split('\n\n')
        self._parsed_sents = self._parse_conll()

        # for each _ means for each list_of_tuples
        self._X_y_split_sents = [[*zip(*_)] for _ in self._parsed_sents]
        self._extract_uniques()

        # Save object to data/interim
        self._pickle()

    def _parse_conll(self) -> List[list]:
        p = re.compile(r'(?:(\S+)(?:\t)(\S+))+')
        return [re.findall(p, sentence) for sentence in self._conll_sents]

    def _extract_uniques(self, label: bool = False) -> None:
        for tok, lab in self._X_y_split_sents:
            self._unique_tokens.update(tok)
            self._unique_labels.update(lab)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two lists, X and y, where X contains
         lists of tokens and y lists of the corresponding labels.
         """
        X_, y_ = zip(*self._X_y_split_sents)
        return (
            np.array([np.array(tuple_) for tuple_ in X_]),
            np.array([np.array(tuple_) for tuple_ in y_])
        )

    def as_lists(self) -> Tuple[List, List]:
        """Returns two numpy arrays, X and y, where X contains
        arrays of tokens and y the corresponding labels arrays.
        """
        X_, y_ = zip(*self._X_y_split_sents)
        return (
            [list(tuple_) for tuple_ in X_],
            [list(tuple_) for tuple_ in y_]
        )

    def as_tuples(self) -> List[List[Tuple[str]]]:
        return self._parsed_sents

    def get_vocab(self) -> List[str]:
        return list(self._unique_tokens)

    def get_labels(self) -> List[str]:
        return list(self._unique_labels)

    def _pickle(self) -> None:
        folder_, file_ = os.path.split(self._abspath)
        pickle_file = f"{file_.rsplit('.')[0]}_ConllData.pickle"

        # Check whether the file is stored directly in data
        if os.path.basename(folder_) == 'data':
            path_to_pickle = os.path.join(folder_, pickle_file)

        # If not, we assume that data contains the folders external, interim, processed, raw
        else:
            # Get parent directory (relative parent should be 'data')
            parent_dir, _ = os.path.split(folder_)
            # Partially processed datasets goes in the 'interim' folder
            interim = os.path.join(parent_dir, 'interim')
            path_to_pickle = os.path.join(interim, pickle_file)

        mode = 'wb' if os.path.exists(path_to_pickle) else 'xb'
        with open(path_to_pickle, mode) as f:
            pickle.dump(self, f)


@dataclass
class Vocab:
    """This class uses the Keras Tokenizer class to build a vocabulary.
    Dict[word] = index
    """
    _tokeniser: Tokenizer = field(init=False)
    word2idx: Dict[str, int] = field(init=False, default_factory=dict)
    idx2word: Dict[int, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._tokeniser = Tokenizer(filters='', lower=False, oov_token='<UNK>')
        self.word2idx['<PAD>'] = 0
        self.idx2word[0] = '<PAD>'

    def _update(self) -> None:
        self.word2idx.update(self._tokeniser.word_index)
        self.idx2word.update(self._tokeniser.index_word)

    def add(self, tokens: Union[List[str], str, Dict[str, int]]) -> None:
        """Fit the tokerniser to the given texts or tokens. Triggers a
        recursive call if tokens is not a list of texts (nested or flat)."""
        if isinstance(tokens, str):
            print(f'token type: {type(tokens)}')
            return self.add((tokens,))
        if isinstance(tokens, dict):
            return self.add(tokens.keys())

        self._tokeniser.fit_on_texts(tokens)
        self._update()

    # def pickle(self, target_directory: str) -> None:
    #     save_pickle(self, target_directory)

    def _vectorise(self, texts):
        return self._tokeniser.texts_to_sequences(texts)

    def vectorise(self, texts):
        return self._vectorise(texts)

    def __len__(self) -> int:
        return len(self.word2idx)


@dataclass
class Data:
    # TODO : this entire module is a hot mess right now
    conll: ConllData
    vocab: Vocab
    X: List[List[int]] = field(init=False)
    y: List[List[int]] = field(init=False)
    padded: bool = field(init=False, default=False)

    Dataset: tf.data.Dataset = field(init=False)

    def __post_init__(self):
        X_, y_ = (self.conll.as_lists())
        self.X = self.vocab.vectorise(X_)
        self.y = self._vectorise_labels(y_)

    def _vectorise_labels(self, y_):
        labels = self.conll.get_labels()
        # reserve 0 for padding
        label2idx = dict(zip(labels, np.arange(1, len(labels) + 1)))

        return [[label2idx[_] for _ in sequence]for sequence in y_]

    def pad_all(self, maxlen=55):
        self.padded = True
        self.X = pad_sequences(self.X, maxlen=maxlen)
        return self.X

    def as_dataset(self) -> tf.data.Dataset:
        X_ = tf.data.Dataset.from_tensor_slices(self.X)
        y_ = tf.data.Dataset.from_tensor_slices(
            list(map(lambda x: to_categorical(x), self.y))
        )
        dataset = tf.data.Dataset.zip((X_, y_))

        self.Dataset = (
            dataset if self.padded
            else dataset.padded_batch()
        )
        return self.Dataset

    def __len__(self):
        return len(self.X)
