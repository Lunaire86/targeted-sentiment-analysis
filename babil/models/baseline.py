#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from argparse import Namespace
from logging import Logger

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Masking
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data.preprocessing import ConllData, Dataset, LabelTokeniser, WordTokeniser
from features.embeddings import load_embeddings
from sandbox.data_generator import Minimal
from utils.config import ArgParser, PathTracker, set_global_seed
from utils.helpers import save_pickle


def pickle_me_elmo(embeddings, vocab, path_to, activate=False):
    # Datasets get pickled automatically, but we
    # need to pickle vocab and embeddings manually
    if activate:
        print('Pickling embeddings and vocab...')
        save_pickle(embeddings, path_to.models)
        save_pickle(vocab, path_to.data)
        print('Done!')


def run(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    args = parsed_args
    path_to = paths

    # Load datasets
    train = ConllData(path_to.train)
    dev = ConllData(path_to.dev)
    test = ConllData(path_to.test)  # Keep ur hands off, bro

    # Load embeddings
    print('Loading pre-trained embeddings...')
    embeddings = np.load(os.path.join(path_to.models, 'small-vec.npy'))
    # embeddings = load_embeddings(args.embedding_id, path_to)
    # small_embeddings = embeddings.__dummy__()
    print(f'Embeddings shape: {embeddings.shape}')
    # print(f', dummy shape: {small_embeddings.shape}')

    # Create shared vocabulary for tasks
    print('Building the vocabulary...')
    vocab = None
    with open(os.path.join(path_to.data, 'Vocab.pickle'), 'rb') as f:
        vocab = pickle.load(f)
    # vocab = Vocab()
    labels = train.get_labels()

    # Add words from both word embeddings and our training data
    # vocab.add(embeddings.vocab)
    # vocab.add(train.get_vocab())
    print('Done!')

    # Convert from ConllData to Dataset
    train_ds = Data(train, vocab)
    dev_ds = Data(dev, vocab)

    pickle_me_elmo(embeddings, vocab, path_to)


    # Create the embedding layer
    # embedding_layer = Embedding(
    #     input_dim=embeddings.vocab_size,     # vocab size
    #     output_dim=embeddings.dim,    # embedding dim
    #     input_length=sequence_length,
    #     weights=[embeddings.weights],
    #     mask_zero=True,
    #     trainable=False,
    # )

    # Create an Input layer
    # input_ = Input(shape=(sequence_length,), name='input')
    # x = embedding_layer(input_)
    # x = Dropout(0.3)(x)
    # x = Bidirectional(LSTM(args.hidden_dim, return_sequences=True))(x)
    # x = Bidirectional(LSTM(32))(x)
    # # x = LSTM(args.hidden_dim)(x)
    # x = Dense(64, activation='relu')(x)
    # output = Dense(len(labels), activation='softmax')(x)


if __name__ == '__main__':
    from babil.__main__ import dev_mode
    args, path_to, logger = dev_mode()

    # Load datasets
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    test = Dataset(path_to.test)  # Keep ur hands off, bro

    # Create tokenisers
    label_tokeniser = LabelTokeniser()
    word_tokeniser = WordTokeniser()

    # Fit to features and labels
    word_tokeniser.fit_on_texts(train.X)
    label_tokeniser.fit_on_texts(train.y)

    # Save them as serialised objects using pickle
    word_tokeniser.save(path_to.interim_data)
    label_tokeniser.save(path_to.interim_data)

    # TODO : start here - build shared vocab
    ...
    # Load embeddings
    # print('Loading pre-trained embeddings...')
    # embeddings = np.load(os.path.join(path_to.models, 'small-vec.npy'))
    # # embeddings = load_embeddings(args.embedding_id, path_to)
    # # small_embeddings = embeddings.__dummy__()
    # print(f'Embeddings shape: {embeddings.shape}')
    # # print(f', dummy shape: {small_embeddings.shape}')
    #
    # # Create shared vocabulary for tasks
    # print('Building the vocabulary...')
    # vocab = None
    # with open(os.path.join(path_to.data, 'Vocab.pickle'), 'rb') as f:
    #     vocab = pickle.load(f)
    # # vocab = Vocab()
    # labels = train.get_labels()
    #
    # # Add words from both word embeddings and our training data
    # # vocab.add(embeddings.vocab)
    # # vocab.add(train.get_vocab())
    # print('Done!')
    #
    # # Convert from ConllData to Dataset
    # train_ds = Data(train, vocab)
    # dev_ds = Data(dev, vocab)
    #
    # pickle_me_elmo(embeddings, vocab, path_to)
    #
    # # some short sentences to test with
    # idx = [0, 2, 14, 26, 47, 48, 72, 77, 78, 79, 80, 83, 84, 85, 85]
    # lil_X = [train_ds.X[i] for i in idx]
    # lil_y = [train_ds.y[i] for i in idx]
    # mini = Minimal(lil_X, lil_y, batch_size=3)
    # ds = tf.data.Dataset.from_generator(mini, tf.int64)
    # for batch in ds:
    #     print(type(batch))
    #     print(f'X shape: {batch[0].shape}\n'
    #           f'y shape: {batch[1].shape}\n')
    #     print(f'First item X: {batch[0[0]]}\n'
    #           f'First item y: {batch[1[0]]}')

    # Create the embedding layer
    # embedding_layer = Embedding(
    #     input_dim=embeddings.vocab_size,     # vocab size
    #     output_dim=embeddings.dim,    # embedding dim
    #     input_length=sequence_length,
    #     weights=[embeddings.weights],
    #     mask_zero=True,
    #     trainable=False,
    # )

    # Create an Input layer
    # input_ = Input(shape=(sequence_length,), name='input')
    # x = embedding_layer(input_)
    # x = Dropout(0.3)(x)
    # x = Bidirectional(LSTM(args.hidden_dim, return_sequences=True))(x)
    # x = Bidirectional(LSTM(32))(x)
    # # x = LSTM(args.hidden_dim)(x)
    # x = Dense(64, activation='relu')(x)
    # output = Dense(len(labels), activation='softmax')(x)