#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
import time
from argparse import Namespace
from os.path import join

from fasttext.FastText import load_model as load_using_fasttext
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from gensim.models import FastText
from gensim.utils import _pickle
from gensim.models.fasttext import FastTextKeyedVectors
from tensorflow.keras.models import Model

from data.preprocessing import Dataset, LabelTokeniser, WordTokeniser, vectorise
from features.embeddings import load_fasttext_embeddings
from models.baseline import Baseline
from utils.config import PathTracker, ArgParser
from utils.helpers import create_dir
from utils.metrics import Metrics


EMBEDDINGS = {
    'cc.no.300.bin': 'fastText Continuous Bag-of-Words: Common Crawl and Wikipedia, dim=300',
    'norec.100.cbow.bin': 'fastText Continuous Bag-of-Words: NoReC, dim=100',
    'norec.300.cbow.bin': 'fastText Continuous Bag-of-Words: NoReC, dim=300',
    'norec.100.sg.bin': 'fastText Skipgram: NoReC, dim=100',
    'norec.300.sg.bin': 'fastText Skipgram: NoReC, dim=300',
    '114.zip': 'fastText Continuous Bag-of-Words: NNC, dim=100',
    '122.zip': 'fastText Skipgram: NNC, dim=100',
    '112.zip': 'fastText Skipgram: NNC + NoWaC, dim=100',
    '120.zip': 'fastText Skipgram: NNC + NoWaC, dim=100',
    '110.zip': 'fastText Continuous Bag-of-Words: NNC + NoWaC + NBDigital, dim=100',
}

# Based on checking the length of all sentences in train and dev,
# setting max sequence length to 50 seems reasonable
SEQUENCE_LENGTH = 50


def baseline(parsed_args: Namespace, paths: PathTracker):
    # Parse and store command line arguments.
    args = parsed_args

    # Read path configuration from json.
    path_to = paths

    # We will either load the model itself, or just a KeyedVectors object.
    # The latter is more practical if we're not going to continue training.
    embeddings: FastTextKeyedVectors

    path_to_embeddings = (
        join(path_to.shared_embeddings, args.embeddings)
        if args.embeddings.endswith('.zip')
        else join(path_to.embeddings, args.embeddings)
    )
    print(f'Loading embeddings: {EMBEDDINGS[args.embeddings]})\n')

    # We have to take a few extra steps if we want to continue training.
    if args.train_embeddings:
        raise NotImplementedError('This feature is not available at the moment.')

    # Load pre-trained word embeddings using Gensim's KeyedVectors format
    try:
        embeddings = load_fasttext_embeddings(path_to_embeddings, args.embeddings)
    except _pickle.UnpicklingError as e:
        print(f'Error caught: {e}')
        embeddings = load_using_fasttext(path_to_embeddings)

    # Get the weights and add vectors representing
    # indices 0 and 1 by concatenating along the first
    # axis (i.e. 'inserting' two rows at the front)
    dim = args.embeddings_dim
    pad_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    unk_vec = tf.random.uniform(shape=[1, dim], minval=0.05, maxval=0.95, seed=69686)
    weights = np.r_[pad_vec.numpy(), unk_vec.numpy(), embeddings.vectors]

    print('Building the vocabulary...\n')
    # Update the vocab with tokens for padding and unknown words
    embeddings_vocab = embeddings.index2entity
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word2idx.update(
        {word: idx + 2 for (idx, word) in enumerate(embeddings_vocab)}
    )
    # idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Create a unique directory where files from this runtime will be stored.
    UNIQUE_DIR = create_dir(path_to.models, args.run)

    print(f'Any files saved during this run will be stored to '
          f'a directory unique to this run: {UNIQUE_DIR}\n')

    # Save the word-to-index dict
    w2i_path = join(UNIQUE_DIR, 'word2idx.pickle')
    mode = 'wb' if os.path.exists(w2i_path) else 'xb'
    with open(w2i_path, mode) as f:
        pickle.dump(word2idx, f)

    # And also save the weights using numpy
    # npy_path = join(UNIQUE_DIR, 'weights.npy')
    # mode = 'wb' if os.path.exists(npy_path) else 'xb'
    # with open(npy_path, mode) as f:
    np.save(join(UNIQUE_DIR, 'embedding_weights.npy'), weights)

    print('Reading data from conll...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)

    # Tokenise words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(train.X)
    word_tokeniser.save(UNIQUE_DIR)
    print(f'Found {len(word_tokeniser.word_index)} different words.')

    # Tokenise labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    label_tokeniser.save(UNIQUE_DIR)  # do we actually need to save this too?
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser, categorical=True)
    y_dev = vectorise(dev.y, label_tokeniser, categorical=True)

    print(f'Train data shapes: X={X_train.shape}, y={y_train.shape}')  # X=(5915, 30) y=(5915, 30, 6)
    print(f'Dev data shapes: X={X_dev.shape}, y={y_dev.shape}')  # X=(1151, 30) y=(1151, 30, 6)

    # Build and train the baseline model
    bilstm = Baseline(args, UNIQUE_DIR, weights)
    bilstm.build()
    bilstm.train(X_train, y_train, X_dev, y_dev)
    # The model gets saved automatically since early stopping is implemented

    # Get predictions
    predictions = bilstm.predict(X_dev)

    # Save predictions and y_dev so we can load for evaluation if needs be
    np.save(join(UNIQUE_DIR, 'y_pred.npy'), predictions)
    np.save(join(UNIQUE_DIR, 'y_gold.npy'), y_dev)

    metrics = Metrics(X_dev, y_dev, predictions, label_tokeniser.index_word)

    # metrics_path = join(path_to.data, f'{args.embeddings}_metrics.pickle')
    # mode = 'wb' if os.path.exists(metrics_path) else 'xb'
    # with open(metrics_path, mode) as f:
    #     pickle.dump(metrics, f)

    print(f'\n\nBinary scores:\n{metrics.binary}')
    print(f'\n\nProportional scores scores:\n{metrics.regular}\n\n')
    print(f'\n\nClassification report for the dev set (excluding "O"):\n{metrics.report}')
    print(f'\n\nClassification report for the dev set (including "O"):\n{metrics.report_including_majority}')

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


if __name__ == '__main__':
    SEQUENCE_LENGTH = 50

    parser = ArgParser()
    args = parser.args

    path_to = (
        PathTracker.from_json('saga_config.json') if args.saga
        else PathTracker.from_json('local_config.json')
    )

    print('Reading data from conll...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)

    # print('Loading pre-trained embeddings...')
    # npy_path = join(path_to.models, 'cc.no.20.bin_weights.npy')
    w2i_path = join(path_to.models, 'cc.no.20.bin_word2idx.pickle')
    pickle_rick = open(w2i_path, 'rb')

    # weights = np.load(npy_path)
    word2idx = pickle.load(pickle_rick)
    pickle_rick.close()

    # basename = 'cc.no.20.bin'
    # path_to_embeddings = join(path_to.models, basename)
    # embeddings = load_facebook_vectors(path_to_embeddings, encoding='latin1')
    #
    # embedding_dim = int(basename.split('.')[2])
    # pad_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    # unk_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    # weights = np.r_[pad_vec.numpy(), unk_vec.numpy(), embeddings.vectors]

    # Update the vocab with tokens for padding and unknown words
    # embeddings_vocab = embeddings.index2entity
    # word2idx = {'<PAD>': 0, '<UNK>': 1}
    # word2idx.update(
    #     {word: idx + 2 for (idx, word) in enumerate(embeddings_vocab)}
    # )
    # idx2word = {idx: word for (word, idx) in word2idx.items()}

    print('Building the vocabulary...')
    # Words
    word_tokeniser = WordTokeniser()
    word_tokeniser.fit_on_texts(train.X)
    word_tokeniser.save(path_to.interim_data)
    print(f'Found {len(word_tokeniser.word_index)} different words.')

    # Labels
    label_tokeniser = LabelTokeniser()
    label_tokeniser.fit_on_texts(train.y)
    label_tokeniser.save(path_to.interim_data)
    num_classes = len(label_tokeniser.word_index)
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser, categorical=True)
    y_dev = vectorise(dev.y, label_tokeniser, categorical=True)

    print(f'Train data shapes: X={X_train.shape}, y={y_train.shape}')  # X=(5915, 30) y=(5915, 30, 6)
    print(f'Dev data shapes: X={X_dev.shape}, y={y_dev.shape}')  # X=(1151, 30) y=(1151, 30, 6)

    smol_mod = join(path_to.models, 'baseline_checkpoint.20.h5')
    model: Model = keras.models.load_model(smol_mod)
    predictions = model.predict(X_dev, batch_size=1)

    # plot_results(model.results, path_to.figures, 'smol', 'accuracy')
    # plot_results(model.results, path_to.figures, 'smol', 'binary_accuracy')
    # plot_results(model.results, path_to.figures, 'smol', 'loss')
    # plot_results(model.results, path_to.figures, 'smol', 'precision')
    # plot_results(model.results, path_to.figures, 'smol', 'recall')
    #
    # # Let's just save these dicts too, while we're at it
    # json_pred = join(path_to.processed_data, f'y_pred_baseline_{basename}.json')
    # json_gold = join(path_to.processed_data, f'y_gold_baseline_{basename}.json')
    # kwargs = {'ensure_ascii': False, 'indent': '\t'}
    #
    # # We'll assume either both files exist, or none
    # mode = 'w' if os.path.exists(json_pred) else 'x'
    #
    # f_p, f_g = open(json_pred, mode), open(json_gold, mode)
    # json.dump(serialise(y_pred), f_p, **kwargs); f_p.close()
    # json.dump(serialise(y_gold), f_g, **kwargs); f_g.close()



    # if args.load:
    #     raise NotImplementedError('This feature is not available at the moment.')

        # TODO : refactor this to make it work after the UNIQUE_DIR implementation
        # Load checkpoint model from previous run
        # bilstm = Baseline.load(path_to.models)
        # bilstm = keras.models.load_model(join(path_to.models, 'baseline_checkpoint.20.h5'))
        # predictions = bilstm.predict(X_dev)