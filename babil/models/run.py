#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from argparse import Namespace
from logging import Logger
from os.path import join
from time import time
from typing import Union

import numpy as np
import tensorflow as tf
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from data.preprocessing import Dataset, LabelTokeniser, WordTokeniser, vectorise
from features import EMBEDDINGS
from features.embeddings import load_gensim_model, load_embeddings
from models.baseline import Baseline
from utils.config import PathTracker
from utils.helpers import get_identifier, path_prefix
from utils.metrics import Metrics


# Based on checking the length of all sentences in train and dev,
# setting max sequence length to 50 seems reasonable
SEQUENCE_LENGTH = 50


def baseline(parsed_args: Namespace, paths: PathTracker, logger: Logger):
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
        embeddings = load_embeddings(path_to_embeddings, args.embeddings)
    except RuntimeError as e:
        logger.error(e)
        embeddings = load_gensim_model(path_to_embeddings, ext='txt')
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
    bilstm = Baseline(args, partial_path, weights)
    bilstm.build()
    logger.info('Training the model...')
    t = time()
    bilstm.train(X_train, y_train, X_dev, y_dev)
    logger.info(f'Training finished after ~{int((time() - t) / 60)} minutes.')
    # The model gets saved automatically since early stopping is implemented

    # Get predictions
    predictions = bilstm.predict(X_dev)

    # Save predictions and y_dev so we can load for evaluation if needs be
    yp_path, yg_path = f'{partial_path}y_pred.npy', f'{partial_path}y_gold.npy'
    np.save(yp_path, predictions)
    np.save(yg_path, y_dev)
    logger.info(f'File saved: {yp_path}')
    logger.info(f'File saved: {yg_path}\n')

    metrics = Metrics(X_dev, y_dev, predictions, label_tokeniser.index_word)

    metrics_output = [
        f'Binary scores: \t{metrics.binary}',
        f'Proportional scores scores:\t{metrics.regular}\n',
        f'Classification report for the dev set (excluding "O"):\t{metrics.report}',
        f'Classification report for the dev set (including "O"):\t{metrics.report_including_majority}'
    ]
    logger.log(f'{_ID_} ~~~ REPORT ~~~')
    print(*metrics_output, sep='\n')
    for m in metrics_output:
        logger.info(m)


    # plot_results(bilstm.results, path_to.figures, 'baseline', 'accuracy')
    # plot_results(bilstm.results, path_to.figures, 'baseline', 'binary_accuracy')
    # plot_results(bilstm.results, path_to.figures, 'baseline', 'loss')
    # plot_results(bilstm.results, path_to.figures, 'baseline', 'precision')
    # plot_results(bilstm.results, path_to.figures, 'baseline', 'recall')


def baseline_eval(parsed_args: Namespace, paths: PathTracker):
    # Parse and store command line arguments.
    args = parsed_args

    # Read path configuration from json.
    path_to = paths

    # Load test data
    test = Dataset(path_to.test)


def improved(args, paths):
    return None
