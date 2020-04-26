#!/usr/bin/env python3
# coding: utf-8
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf

from babil.data.preprocessing import ConllData, Vocab
from babil.features.embeddings import WordEmbeddings

if __name__ == '__main__':
    # Store command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_layers', '-nl', action='store', type=int, default=1, help='Number of layer(s)')
    parser.add_argument('--hidden_dim', '-hd', action='store', type=int, default=100,
                        help='Size of the hidden layer(s)')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=50, help='Size of mini-batches')
    parser.add_argument('--dropout', '-dr', action='store', type=float, default=0.01)
    parser.add_argument('--learning_rate', '-lr', action='store', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', '-e', action='store',
                        help='Early stopping is implemented, so this may be an arbitrary large number', type=int,
                        default=50)
    parser.add_argument('--embedding_id', '-eid', action='store', type=int, default=200,
                        help='Word embedding ID number (as per http://vectors.nlpl.eu)')
    parser.add_argument('--embedding_dim', '-edim', action='store', type=int, default=100,
                        help='Dimensionality of the word embeddings')
    parser.add_argument('--train_embeddings', '-tre', action='store_true', help='Trainable embeddings?')
    parser.add_argument('-S', '--saga', action='store_true', help='Load embeddings from Saga')
    args = parser.parse_args()
    print(args)

    # Ensure reproducibility
    global_seed: int
    with open('../../SEED', 'r') as f:
        global_seed = int(f.read())
    tf.random.set_seed(global_seed)
    print(f"Global seed set: {global_seed}")

    # Set project root dir for easy pickling of objects
    root = '/home/fero/Desktop/nlp/in5550-2020-exam'
    conll = {'train': os.path.join(root, 'data/raw/train.conll'),
             'dev': os.path.join(root, 'data/raw/dev.conll'),
             'test': os.path.join(root, 'data/raw/test.conll')}

    # Load datasets
    train = ConllData(conll['train'])
    dev = ConllData(conll['dev'])
    test = ConllData(conll['test'])

    # Load embeddings
    print('Loading pre-trained embeddings...')
    embeddings = WordEmbeddings(filepath='/media/fero/Programs/nlp/embeddings/norwegian/58.zip')

    # Create shared vocabulary for tasks
    vocab = Vocab()
    labels = train.get_labels()

    # Add words from both word embeddings and our training data
    vocab.add(embeddings.vocab)
    vocab.add(train.get_vocab())

    # Datasets get pickled automatically, but we
    # need to pickle vocab and embeddings manually
    embeddings.pickle(os.path.join(root, 'models/embeddings/norwegian'))
    vocab.pickle(os.path.join(root, 'data/interim'))

    # Create shared vocabulary for tasks
    ...
