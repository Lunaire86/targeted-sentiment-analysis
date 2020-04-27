#!/bin/env python3
# coding: utf-8

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import List, Dict, Union
from typing import Tuple, Set

from tensorflow.keras.preprocessing.text import Tokenizer

from babil.utils.helpers import pickle


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

    def as_lists(self) -> List[List[Tuple[str]]]:
        return self._X_y_split_sents

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

    def vectorise(self, texts):
        return self._vectorise(texts)

    def pickle(self, target_directory: str) -> None:
        pickle(self, target_directory)

    def _vectorise(self, texts):
        return self._tokeniser.texts_to_sequences(texts)

    def __len__(self) -> int:
        return len(self.word2idx)
