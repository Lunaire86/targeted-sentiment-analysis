#!/usr/bin/env python3
# coding: utf-8
import os
import pickle
import zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Any

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

from babil import set_global_seed
from utils.config import PathTracker


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
    """Supports different file formats. Uses the Gensim library.
    Creates an embedding matrix and two dictionaries
    (1) a word to index dictionary which returns the index
    in the embedding matrix
    (2) an index to word dictionary which returns the word
    given an index.
    """
    filepath: str
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
        self._model = load_gensim_model(self.filepath)
        self._vocab = self._model.index2entity
        self._vectors = self._model.vectors
        self.dim = self._vectors.shape[1]
        self._init_vocab()
        self._set_weights()

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


def _lookup(identifier, topdir) -> Optional[str]:
    """Search folders and subfolders, starting at`topdir`.
    Return an absolute path if a file is found, else None.
    """
    # These file-types cover the most likely candidates for word embeddings
    filetypes = ['txt', 'txt.gz', 'zip', 'bin', 'vec', 'vec.gz', 'pickle']
    options = [f'{identifier}.{suffix}' for suffix in filetypes]
    # This adds support for pickle files we may have lying around from earlier run
    options.append(f'WordEmbeddings_{identifier}.pickle')

    for path, _, files, _ in os.fwalk(topdir):
        match = set(options).intersection(files)

        # If we have a match (or more), we return it by popping either
        # the only element, or an arbitrary one — either is fine.
        if match:
            abspath = os.path.join(path, match.pop())
            return abspath if os.path.isfile(abspath) else None
    return None


def load_embeddings(identifier: Union[str, int], path_to: Optional[PathTracker] = '') -> WordEmbeddings:
    """Given an identifier, whether it is an actual file name or a
    numerical identifier, look for word embeddings and load if found.
    `identifier` can be an integer (ID), or the name of the file,
    with or without the file-type attached. The identifiers 200, '200'
    and '200.zip' will all result in the same word embeddings being loaded.

    Unless `identifier` is an absolute path, a PathTo object must be provided.
    """
    # First, check if `identifier` is actually the full path to a file
    if os.path.isfile(identifier):
        if identifier.endswith('.pickle'):
            with open(identifier) as f:
                return pickle.load(f)
        else:
            return WordEmbeddings(identifier)

    # Search project directory first. If we have pickled
    # embeddings from previous runs, we'd like to find them first
    path_found = _lookup(identifier, path_to.project_root) or _lookup(identifier, path_to.embeddings)

    if path_found:
        return WordEmbeddings(path_found)
    raise FileNotFoundError()
