#!/usr/bin/env python3
# coding: utf-8

import pickle
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

from babil import set_global_seed


def load_gensim_model(filepath: str) -> KeyedVectors:
    """Detect the model format by its extension."""
    model: KeyedVectors
    is_binary = False if filepath.endswith(('.txt.gz', '.txt', '.vec.gz', '.vec')) else True
    kwargs: Dict = {'binary': is_binary,
                    'unicode_errors': 'replace'}

    # ZIP archive from the NLPL vector repository:
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, "r") as archive:
            model = KeyedVectors.load_word2vec_format(
                archive.open("model.bin"), **kwargs)
    else:
        model = KeyedVectors.load_word2vec_format(
            filepath, **kwargs)

    # Unit-normalizing the vectors (if they aren't already)
    model.init_sims(replace=True)
    return model


@dataclass
class WordEmbeddings:
    """Import word2vec files saved in txt format.
    Creates an embedding matrix and two dictionaries
    (1) a word to index dictionary which returns the index
    in the embedding matrix
    (2) an index to word dictionary which returns the word
    given an index.
    """
    filepath: str
    file_type: str = 'word2vec'
    encoding: str = 'utf8'
    pad: bool = True
    unk: bool = True
    vocab: Union[List, Dict[str, int]] = field(default_factory=list)

    vocab_size: int = field(init=False)
    dim: int = field(init=False)
    weights: np.ndarray = field(init=False)
    word2idx: Dict[str, int] = field(init=False)
    idx2word: Dict[int, str] = field(init=False)
    _model: KeyedVectors = field(init=False)

    _vectors: np.ndarray = field(init=False)
    _vocab: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        set_global_seed()
        if self.file_type is 'word2vec':
            self._model = load_gensim_model(self.filepath)
            self._vocab = self._model.index2entity
            self._vectors = self._model.vectors
            self.dim = self._vectors.shape[1]
            self._init_vocab()
            self._set_weights()
        else:
            raise NotImplementedError(f'{self.file_type} is not supported yet!')

    def _init_vocab(self) -> None:
        # If the call to __init__() did not include an external vocabulary,
        # we will set vocab equal to the embedding's vocabulary.
        self.vocab = self.vocab or self._vocab
        self.vocab_size = len(self.vocab)

        self.word2idx = self._model.vocab
        self.idx2word = self._model.index2word

    def _set_weights(self) -> None:
        if self.pad and self.unk:
            # Add vectors representing indices 0 and 1 by concatenating
            # along the first axis (i.e. 'inserting' two rows at the front)
            self.weights = np.r_[tf.random.uniform((2, self.dim)), self._vectors]
        elif self.pad or self.unk:
            self.weights = np.r_[tf.random.uniform((1, self.dim)), self._vectors]
        else:
            self.weights = self._vectors

    # def save_pickle(self, target_directory: str) -> None:
    #     pickle(self, target_directory)
    #     # folder_ = os.path.abspath(path_to_folder)
        # embedding_id = os.path.basename(self.filepath).rsplit('.')[0]
        # pickle_file = embedding_id + '_WordEmbeddings.pickle'
        # path_to_pickle = os.path.join(folder_, pickle_file)
        #
        # mode = 'wb' if os.path.exists(path_to_pickle) else 'xb'
        # with open(path_to_pickle, mode) as f:
        #     pickle.dump(self, f)
