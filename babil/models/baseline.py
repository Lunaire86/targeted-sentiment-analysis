#!/usr/bin/env python3
# coding: utf-8

import os
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.layers import Input, LSTM, LSTMCell, Masking
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adamax, Adam

from babil import set_global_seed
from data.preprocessing import ConllData, Vocab
from features.embeddings import WordEmbeddings
from utils import ArgParser
from utils.helpers import generate_path, open_pickle, save_pickle


if __name__ == '__main__':
    # Store command line arguments
    parser = ArgParser()    # class defined in utils.ArgParser
    args = parser.args
    print(args)

    # Ensure reproducibility
    set_global_seed()

    # Set project root dir for easy pickling of objects
    root = '/home/fero/Desktop/nlp/in5550-2020-exam'
    conll = {'train': os.path.join(root, 'data/raw/train.conll'),
             'dev': os.path.join(root, 'data/raw/dev.conll'),
             'test': os.path.join(root, 'data/raw/test.conll')}

    # Load datasets
    train = ConllData(conll['train'])
    dev = ConllData(conll['dev'])
    test = ConllData(conll['test'])  # Keep ur hands off, bro

    # Load embeddings
    print('Loading pre-trained embeddings...')
    embedding_path = generate_path(location='local', identifier=args.embedding_id)
    embeddings = open_pickle('embeddings', args.embedding_id) or \
                 WordEmbeddings(filepath='/media/fero/Programs/nlp/embeddings/norwegian/58.zip')

    pickled_embeddings = f'WordEmbeddings_{args.embedding_id}.pickle'
    if pickled_embeddings in os.listdir(os.path.walk(root)):
        embeddings = pickle.load(pickled_embeddings)
    else:
        embeddings = WordEmbeddings(filepath='/media/fero/Programs/nlp/embeddings/norwegian/58.zip')
    print('Done!')

    # Create shared vocabulary for tasks
    print('Building the vocabulary...')
    vocab = Vocab()
    labels = train.get_labels()

    # Add words from both word embeddings and our training data
    vocab.add(embeddings.vocab)
    vocab.add(train.get_vocab())
    print('Done!')

    '''
    train_tokens = train_df['form_vec'].apply(np.ravel)
    train_labels = train_df['upos_vec'].apply(np.ravel)
    dev_tokens = dev_df['form_vec'].apply(np.ravel)
    dev_labels = dev_df['upos_vec'].apply(np.ravel)



    # Vectorise by mapping tokens to corresponding embedding
    # indices, then pad X_train and X_test to the same length
    X_train = pad_sequences(train_tokens, maxlen=sequence_length)
    y_train = pad_sequences(train_labels, maxlen=sequence_length)

    X_test = pad_sequences(dev_tokens, maxlen=sequence_length)
    y_test = pad_sequences(dev_labels, maxlen=sequence_length)
    '''
    # Find the length of the longest sequence
    # sequence_length = max(max(train.as_lists().apply(len)),
    #                       max(dev_tokens.apply(len)))

    # Pad sequences
    sequence_length = 50

    # Datasets get pickled automatically, but we
    # need to pickle vocab and embeddings manually
    print('Pickling embeddings and vocab...')
    save_pickle(embeddings, os.path.join(root, 'models'))
    save_pickle(vocab, os.path.join(root, 'data/interim'))
    print('Done!')
    # embeddings.pickle(os.path.join(root, 'models/embeddings/norwegian'))
    # vocab.pickle(os.path.join(root, 'data/interim'))

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
