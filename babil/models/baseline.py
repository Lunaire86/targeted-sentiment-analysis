#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
from argparse import Namespace
from logging import Logger
from os.path import join
from typing import Union

import numpy as np
import tensorflow as tf
from gensim.models import FastText
from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_model, load_facebook_vectors

from data.preprocessing import Dataset, LabelTokeniser, WordTokeniser, vectorise
from utils.config import PathTracker


def throwaway():
    pass


def run(parsed_args: Namespace, paths: PathTracker, logger: Logger):
    # Based on checking the length of all sentences in train and dev,
    # setting max sequence length to 50 seems reasonable
    MAX_SEQUENCE_LENGTH = 50

    args = parsed_args
    path_to = paths

    print('Loading the dataset...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    test = Dataset(path_to.test)  # Keep ur hands off, bro

    print('Loading pre-trained embeddings...')
    basename: str = ''
    embeddings: Union[FastText, FastTextKeyedVectors] = None
    if args.saga:
        basename = 'cc.no.300.bin'
        path_to_embeddings = join(path_to.models, basename)  # saga
        embeddings = load_facebook_model(path_to_embeddings, encoding='latin1')
    else:
        basename = 'cc.no.20.bin'
        path_to_embeddings = join(path_to.embeddings, 'norwegian', basename)
        embeddings = load_facebook_vectors(path_to_embeddings, encoding='latin1')

    # Get the weights and add vectors representing
    # indices 0 and 1 by concatenating along the first
    # axis (i.e. 'inserting' two rows at the front)
    embedding_dim = int(basename.split('.')[2])
    pad_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    unk_vec = tf.random.uniform(shape=[1, embedding_dim], minval=0.05, maxval=0.95, seed=69686)
    weights = np.r_[pad_vec.numpy(), unk_vec.numpy(), embeddings.vectors]

    # Update the vocab with tokens for padding and unknown words
    embeddings_vocab = embeddings.index2entity
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    word2idx.update(
        {word: idx + 2 for (idx, word) in enumerate(embeddings_vocab)}
    )
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Save the word-to-index dict
    w2i_path = join(path_to.models, f'{basename}_word2idx.pickle')
    mode = 'wb' if os.path.exists(w2i_path) else 'xb'
    with open(w2i_path, mode) as f:
        pickle.dump(word2idx, f)

    # And also save the weights using numpy
    npy_path = join(path_to.models, f'{basename}_weights.npy')
    mode = 'wb' if os.path.exists(npy_path) else 'xb'
    with open(npy_path, mode) as f:
        np.save(f, weights)

    print('Building the vocabulary...')
    # Create tokenisers
    label_tokeniser = LabelTokeniser()
    word_tokeniser = WordTokeniser()

    # Fit to features and labels
    word_tokeniser.fit_on_texts(train.X)
    label_tokeniser.fit_on_texts(train.y)

    # Save them as serialised objects using pickle
    word_tokeniser.save(path_to.interim_data)
    label_tokeniser.save(path_to.interim_data)

    X_train = ''
    X_dev = ''

    y_train = ''
    y_dev = ''

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
    # Based on checking the length of all sentences in train and dev,
    # setting max sequence length to 50 seems reasonable
    MAX_SEQUENCE_LENGTH = 50

    # Setup for running locally, as opposed to on Saga
    from babil.__main__ import dev_mode

    args, path_to, logger = dev_mode()

    print('Loading the dataset...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    test = Dataset(path_to.test)  # Keep ur hands off, bro

    # We will just load the pickled files here
    # See run() for implementation of loading the actual model
    print('Loading pre-trained embeddings...')
    npy_path = join(path_to.models, 'cc.no.20.bin_weights.npy')
    w2i_path = join(path_to.models, 'cc.no.20.bin_word2idx.pickle')
    f = open(w2i_path, 'rb')

    weights = np.load(npy_path)
    word2idx = pickle.load(f)

    idx2word = {idx: word for (word, idx) in word2idx.items()}
    f.close()

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
    print(f'Found {len(label_tokeniser.word_index)} different labels.')

    # Vectorise words
    X_train = vectorise(train.X, word2idx=word2idx)
    X_dev = vectorise(dev.X, word2idx=word2idx)

    # Vectorise labels
    y_train = vectorise(train.y, label_tokeniser, categorical=True)
    y_dev = vectorise(dev.y, label_tokeniser, categorical=True)

    # Create shared vocabulary for tasks
    # vocab = None
    # with open(os.path.join(path_to.data, 'Vocab.pickle'), 'rb') as f:
    #     vocab = pickle.load(f)
    # # vocab = Vocab()
    # labels = train.get_labels()

    # Load embeddings
    # print('Loading pre-trained embeddings...')
    # embeddings = np.load(os.path.join(path_to.models, 'small-vec.npy'))
    # # embeddings = load_embeddings(args.embedding_id, path_to)
    # # small_embeddings = embeddings.__dummy__()
    # print(f'Embeddings shape: {embeddings.shape}')
    # # print(f', dummy shape: {small_embeddings.shape}')
    #
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
