#!/usr/bin/env python3
# coding: utf-8

from typing import List, Dict
from zipfile import ZipFile

from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
from gensim.models.fasttext import load_facebook_vectors


def train_embeddings(model: FastText, sentences: List[List[str]]):
    # Update the embeddings with sentences from our training set
    model.build_vocab(
        sentences,
        update=True
    )
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=model.epochs
    )


def load_embeddings(path: str, name: str):
    if name.startswith('cc'):
        # Case: Native fastText embeddings.
        return load_facebook_vectors(path, encoding='latin1')

    if name.endswith('bin'):
        # Case: Models trained specifically for this project.
        model = FastText.load(path)
        # Pre-compute L2-normalized vectors.
        model.init_sims(replace=True)
        return model.wv

    if name.endswith('zip'):
        return load_gensim_model(filepath=path)


def load_gensim_model(filepath: str, ext: str = 'bin') -> KeyedVectors:
    """Detect the model format by its extension."""
    model: KeyedVectors
    is_binary = False if filepath.endswith(('.txt.gz', '.txt', '.vec.gz', '.vec')) else True
    kwargs: Dict = {'binary': is_binary,
                    'encoding': 'latin1',
                    'unicode_errors': 'replace'}

    # ZIP archive from the NLPL vector repository:
    if filepath.endswith('.zip'):
        with ZipFile(filepath, "r") as archive:
            model = KeyedVectors.load_word2vec_format(
                archive.open(f'model.{ext}'), **kwargs)
    else:
        model = KeyedVectors.load_word2vec_format(
            filepath, **kwargs)

    # Unit-normalizing the vectors (if they aren't already)
    model.init_sims(replace=True)
    return model
