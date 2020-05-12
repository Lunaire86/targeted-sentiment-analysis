#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from argparse import Namespace
from logging import Logger
from os.path import join
from time import time
from typing import Union, Any, Tuple, List, Optional, Dict

import numpy as np
import tensorflow as tf
from gensim.models.fasttext import FastTextKeyedVectors, FastText
from gensim.models.keyedvectors import Word2VecKeyedVectors


import features
from coleus.model import FastTextModel
from data.preprocessing import Dataset, LabelTokeniser, WordTokeniser, vectorise
from features import EMBEDDINGS
from features import embeddings
from models import baseline, improved
from utils.config import PathTracker
from utils.helpers import get_identifier, path_prefix, plot_results, flatten
from utils.metrics import Metrics


# Based on checking the length of all sentences in train and dev,
# setting max sequence length to 50 seems reasonable
SEQUENCE_LENGTH = 50


def setup(args: Namespace, path_to: PathTracker, logger: Logger):
    # Create a unique runtime id that we'll use for
    # files saved during this run.
    _ID_ = get_identifier(path_to.models)
    partial_path = path_prefix(path_to.models, _ID_)
    s = f'Any files saved during this run will be prefixed with {_ID_}'
    print(f'\n{s}');
    logger.info(s)

    path_to_embeddings = (
        join(path_to.shared_embeddings, args.embeddings)
        if args.embeddings.endswith('.zip')
        else join(path_to.embeddings, args.embeddings)
    )
    return _ID_, partial_path, path_to_embeddings


def baseline_eval(parsed_args: Namespace, paths: PathTracker):
    # Parse and store command line arguments.
    args = parsed_args

    # Read path configuration from json.
    path_to = paths

    # Load test data
    test = Dataset(path_to.test)


def build_vocab(idx2word: Dict[int, str], logger: Logger):
    s = 'Building the vocabulary...'
    print(s)
    logger.info(s)

    # Update the vocab with tokens for padding and unknown words
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word2idx.update(
        {word: idx + 2 for (idx, word) in enumerate(idx2word)}
    )


def train_model(model: Union[baseline.BiLSTM, improved.BiLSTM],
                X_train, y_train, X_dev, y_dev,
                logger: Logger):

    logger.info('Training the model...')
    t = time()
    model.train(X_train, y_train, X_dev, y_dev)
    logger.info(f'Training finished after ~{int((time() - t) / 60)} minutes.')
    # The model gets saved automatically since early stopping is implemented


def _load_for_training(path: str, name: str, logger) -> FastText:
    # TODO : implement
    raise NotImplementedError('Most unfortunate...')


def load_embeddings(path: str, name: str, train: bool, logger: Logger) -> Optional[Union[FastTextModel, FastTextKeyedVectors, Word2VecKeyedVectors]]:

    model: Optional[Union[FastTextModel, FastTextKeyedVectors, Word2VecKeyedVectors, FastText, Any]] = None
    s = f'Loading embeddings: {EMBEDDINGS[name]}'
    print(s, '\n')
    logger.info(s)

    try:
        model = embeddings.load(path, name)

    except Exception as e:
        logger.error(e)

        try:
            model = FastTextModel.load(path)

        except Exception as e:
            logger.error(e)
            raise RuntimeError('Failed to load embeddings!')

    # model = embeddings.load(path, name) if not train else _load_for_training(path, name, logger)
    # model = embeddings.load_gensim_model(path, ext='txt')
    logger.info(f'Embeddings loaded: {model}')
    return model


def load_dataset(train_path: str,
                 dev_path: str,
                 logger: Logger) -> Tuple[Dataset, Dataset]:

    s = 'Reading data from conll...'
    print(s)
    logger.info(s)
    return Dataset(train_path), Dataset(dev_path)


def get_tokenisers(X_train: List[List[str]],
                   y_train: List[List[str]],
                   logger: Logger
                   ) -> Tuple[WordTokeniser, LabelTokeniser]:

    # Tokenise words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(X_train)

    # Tokenise labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(y_train)

    s = f'Found {len(word_tokeniser.word_index)} words and {len(label_tokeniser.word_index)} labels'
    print(s)
    logger.info(s)

    return word_tokeniser, label_tokeniser


def save(file: str,
         obj: Union[Any, np.ndarray],
         logger: Logger
         ) -> None:

    mode = 'wb' if os.path.exists(file) else 'xb'
    if isinstance(obj, np.ndarray):
        np.save(file, obj)
    else:
        with open(file, mode) as f:
            pickle.dump(file, f)
    logger.info(f'File saved: {file}')


def get_weights(name: str,
                vectors: np.ndarray
                ) -> np.ndarray:

    # Get the weights and add vectors representing
    # indices 0 and 1 by concatenating along the first
    # axis (i.e. 'inserting' two rows at the front)
    dim = int(EMBEDDINGS[name].rsplit('=', 1)[1])
    pad_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    unk_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    return np.r_[pad_vec.numpy(), unk_vec.numpy(), vectors]


def vectorise_dataset(X, y, tokeniser, word2idx):
    return (
        vectorise(X, word2idx=word2idx),
        vectorise(y, tokeniser, categorical=True)
    )


def baseline_trainer(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    # Parse and store command line arguments.
    args = parsed_args

    # Read path configuration from json.
    path_to = paths

    # Create a unique runtime id that we'll use for
    # files saved during this run.
    _ID_ = get_identifier(path_to.models)
    partial_path = path_prefix(path_to.models, _ID_)
    s = f'Any files saved during this run will be prefixed with {_ID_}'
    print(f'\n{s}'); logger.info(s)

    # We will either load the model itself, or just a KeyedVectors object.
    # The latter is more practical if we're not going to continue training.
    embeddings: Union[FastTextKeyedVectors, Word2VecKeyedVectors] = None

    path_to_embeddings = (
        join(path_to.shared_embeddings, args.embeddings)
        if args.embeddings.endswith('.zip')
        else join(path_to.embeddings, args.embeddings)
    )
    s = f'Loading embeddings: {EMBEDDINGS[args.embeddings]}'
    print(s, '\n'); logger.info(s)

    # We have to take a few extra steps if we want to continue training.
    if args.train_embeddings:
        raise NotImplementedError('This feature is not available at the moment.')

    # Load pre-trained word embeddings
    try:
        embeddings = features.embeddings.load(path_to_embeddings, args.embeddings)
    except RuntimeError as e:
        logger.error(e)
        embeddings = features.embeddings.load_gensim_model(path_to_embeddings, ext='txt')
    finally:
        logger.info(f'Embeddings loaded: {embeddings}')

    # Get the weights and add vectors representing
    # indices 0 and 1 by concatenating along the first
    # axis (i.e. 'inserting' two rows at the front)
    dim = int(EMBEDDINGS[args.embeddings].rsplit('=', 1)[1])
    pad_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    unk_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    weights = np.r_[pad_vec.numpy(), unk_vec.numpy(), embeddings.vectors]

    s = 'Building the vocabulary...'
    print(s); logger.info(s)
    # Update the vocab with tokens for padding and unknown words
    embeddings_vocab = embeddings.index2entity
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word2idx.update(
        {word: idx + 2 for (idx, word) in enumerate(embeddings_vocab)}
    )

    # Save the word-to-index dict
    w2i_path = f'{partial_path}_word2idx.pickle'
    mode = 'wb' if os.path.exists(w2i_path) else 'xb'
    with open(w2i_path, mode) as f:
        pickle.dump(word2idx, f)
    logger.info(f'File saved: {w2i_path}')

    # Save the weights too
    npy_path = f'{partial_path}_embeddings_weights.npy'
    np.save(npy_path, weights)
    logger.info(f'File saved: {npy_path}')

    s = 'Reading data from conll...'
    print(s); logger.info(s)
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)

    # Tokenise words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(train.X)
    word_tokeniser.save(partial_path)
    s = f'Found {len(word_tokeniser.word_index)} different words.'
    print(s); logger.info(s)

    # Tokenise labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    label_tokeniser.save(partial_path)  # do we actually need to save this too?
    s = f'Found {len(label_tokeniser.word_index)} different labels.'
    print(s); logger.info(s)

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser, categorical=True)
    y_dev = vectorise(dev.y, label_tokeniser, categorical=True)

    s = f'Train data shapes: X={X_train.shape}, y={y_train.shape}'
    print(s); logger.info(s)  # X=(5915, 30) y=(5915, 30, 6)
    s = f'Dev data shapes: X={X_dev.shape}, y={y_dev.shape}'
    print(s), logger.info(s)  # X=(1151, 30) y=(1151, 30, 6)

    # Build and train the baseline model
    model = baseline_trainer.BiLSTM(args, partial_path, weights)
    model.build()
    logger.info('Training the model...')
    t = time()
    model.train(X_train, y_train, X_dev, y_dev)
    logger.info(f'Training finished after ~{int((time() - t) / 60)} minutes.')
    # The model gets saved automatically since early stopping is implemented

    # Get predictions
    predictions = model.predict(X_dev)

    # Save predictions and y_dev so we can load for evaluation if needs be
    yp_path, yg_path = f'{partial_path}y_pred.npy', f'{partial_path}y_gold.npy'
    np.save(yp_path, predictions)
    np.save(yg_path, y_dev)
    logger.info(f'File saved: {yp_path}')
    logger.info(f'File saved: {yg_path}\n')

    metrics = Metrics(X_dev, y_dev, predictions, label_tokeniser.index_word)

    metrics_output = [
        f'\nBinary scores: \n{metrics.binary}',
        f'\nProportional scores scores:\n{metrics.regular}\n',
        f'Classification report for the dev set (excluding "O"):\n{metrics.report}',
        f'Classification report for the dev set (including "O"):\n{metrics.report_including_majority}'
    ]
    logger.info(f'{_ID_} ~~~ REPORT ~~~')
    print(*metrics_output, sep='\n')
    for m in metrics_output:
        logger.info(m)

    plot_results(model.results, path_to.figures, 'baseline', 'accuracy')
    plot_results(model.results, path_to.figures, 'baseline', 'binary_accuracy')
    plot_results(model.results, path_to.figures, 'baseline', 'loss')
    plot_results(model.results, path_to.figures, 'baseline', 'precision')
    plot_results(model.results, path_to.figures, 'baseline', 'recall')


def print_report(metrics: Metrics, _ID_: str, logger: Logger) -> None:
    metrics_output = [
        f'\nBinary scores: \n{metrics.binary}',
        f'\nProportional scores scores:\n{metrics.regular}\n',
        f'Classification report for the dev set (excluding "O"):\n{metrics.report}',
        f'Classification report for the dev set (including "O"):\n{metrics.report_including_majority}'
    ]

    logger.info(f'{_ID_} ~~~ REPORT ~~~')
    print(*metrics_output, sep='\n')
    for m in metrics_output:
        logger.info(m)


def improved_trainer(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    # Parse and store command line arguments.
    args = parsed_args

    # Read path configuration from json.
    path_to = paths

    # Create a unique runtime id that we'll use for
    # files saved during this run, and define the paths.
    _ID_, partial_path, path_to_embeddings = setup(args, path_to, logger)

    # Load in the datasets for train and dev
    train, dev = load_dataset(path_to.train, path_to.dev, logger)

    # Load pre-trained word embeddings and get adjusted weights
    embeddings = load_embeddings(
        path=path_to_embeddings,
        name=args.embeddings,
        train=args.train_embeddings,
        logger=logger
    )
    weights = get_weights(args.embeddings, embeddings.vectors)

    # Update the vocab index with tokens <PAD> and <UNK>
    word2idx = build_vocab(embeddings.index2entity, logger)

    # Create tokenisers
    word_tokeniser, label_tokeniser = get_tokenisers(train.X, train.y, logger)

    # Vectorise the datasets
    X_train, y_train = vectorise_dataset(
        train.X, train.y,
        tokeniser=label_tokeniser,
        word2idx=word2idx
    )

    X_dev, y_dev = vectorise_dataset(
        dev.X, dev.y,
        tokeniser=label_tokeniser,
        word2idx=word2idx
    )

    sx = f'Train data shapes: X={X_train.shape}, y={y_train.shape}'
    sy = f'Dev data shapes: X={X_dev.shape}, y={y_dev.shape}'
    print(sx, sy, sep='\n')
    logger.info(sx)
    logger.info(sy)

    # Build and train a model
    model = improved.BiLSTM(args, partial_path, weights)
    model.build()
    train_model(model, X_train, y_train, X_dev, y_dev, logger)

    # Get predictions
    predictions = model.predict(X_dev)

    metrics = Metrics(X_dev, y_dev, predictions, label_tokeniser.index_word)
    print_report(metrics, _ID_, logger)

    # Save various objects
    path2obj = {
        f'{partial_path}_word2idx.pickle': word2idx,
        f'{partial_path}_embeddings_weights.npy': weights,
        f'{partial_path}y_pred.npy': predictions,
        f'{partial_path}y_gold.npy': y_dev
    }
    for path, obj in path2obj.items():
        save(path, obj, logger)
    del path2obj
    word_tokeniser.save(partial_path)
    label_tokeniser.save(partial_path)

    # Plot or go home!
    plot_results(model.results, path_to.figures, 'improved', 'accuracy')
    plot_results(model.results, path_to.figures, 'improved', 'binary_accuracy')
    plot_results(model.results, path_to.figures, 'improved', 'loss')
    plot_results(model.results, path_to.figures, 'improved', 'precision')
    plot_results(model.results, path_to.figures, 'improved', 'recall')
